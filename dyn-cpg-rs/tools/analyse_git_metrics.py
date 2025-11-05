#!/usr/bin/env python3

from typing import Callable, List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import matplotlib.colors as mcolors
from matplotlib.ticker import LogLocator, LogFormatterSciNotation
from scipy import stats

import matplotlib

matplotlib.use("pgf")
matplotlib.rcParams.update(
    {
        "pgf.texsystem": "pdflatex",
        "font.family": "serif",
        "image.composite_image": False,
        "text.usetex": True,
        "pgf.rcfonts": False,
        "font.size": 11,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    }
)


# --- Config --- #

sns.set_palette("Set2")
ANALYSES: List[Callable[[pd.DataFrame, Path], None]] = []

# --- Analysis registration --- #


def analysis(func):
    ANALYSES.append(func)
    return func


# --- Helpers --- #


def save_plot(fig, output_file: Path):
    fig.tight_layout()
    fig.patch.set_alpha(0.0)
    fig.savefig(output_file.with_suffix(".pgf"), bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


# --- Analyses --- #


@analysis
def incremental_vs_full_by_edits(df: pd.DataFrame, output_file: Path, hue_cap=25):

    df = df.copy()

    fig, ax = plt.subplots(figsize=(10, 8))
    fig.set_size_inches(w=5.9 / 2, h=5.9 / 2)

    norm = mcolors.LogNorm(vmin=max(df["edits_count"].min(), 1), vmax=hue_cap)
    cmap = plt.get_cmap("viridis")
    colors = cmap(norm(df["edits_count"].to_numpy()))

    ax.scatter(
        df["full_timings_ms"],
        df["incremental_timings_ms"],
        c=colors,
        alpha=0.7,
    )

    # Limits
    lim_min = min(df["full_timings_ms"].min(), df["incremental_timings_ms"].min()) * 0.9
    lim_max = max(df["full_timings_ms"].max(), df["incremental_timings_ms"].max()) * 1.1

    # Highlight top-left: incremental > full
    x_fill = np.linspace(lim_min, lim_max, 500)
    ax.fill_betweenx(x_fill, lim_min, x_fill, color="red", alpha=0.2, zorder=0)

    # y=x reference
    ax.plot([lim_min, lim_max], [lim_min, lim_max], "r--", linewidth=1, label="y=x")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lim_min, lim_max)
    ax.set_ylim(lim_min, lim_max)

    # Set logarithmic tick locators and formatters
    ax.xaxis.set_major_locator(LogLocator(base=10.0, numticks=10))
    ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=10))
    ax.xaxis.set_minor_locator(
        LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=10)
    )
    ax.yaxis.set_minor_locator(
        LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=10)
    )

    ax.set_xlabel("Full Timings (ms)")
    ax.set_ylabel("Incremental Timings (ms)")
    ax.grid(True, linestyle=":", linewidth=0.5, which="major")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(
        sm, ax=ax, label="Edit Counts (capped at {})".format(hue_cap), shrink=0.7
    )
    cbar.solids.set_rasterized(False)

    ax.set_aspect("equal", adjustable="box")

    save_plot(fig, output_file)


@analysis
def incremental_vs_full_heatmap(df: pd.DataFrame, output_file: Path):
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.set_size_inches(w=5.9 / 2, h=5.9 / 2)

    # Log-transform
    x = np.log10(df["full_timings_ms"].clip(lower=0.1))
    y = np.log10(df["incremental_timings_ms"].clip(lower=0.1))

    # Limits
    lim_min = min(x.min(), y.min()) * 1.1
    lim_max = max(x.max(), y.max()) * 1.1

    # Bins
    bin_width = 0.2
    lim_min_bin = np.floor(lim_min / bin_width) * bin_width
    lim_max_bin = np.ceil(lim_max / bin_width) * bin_width
    bins_edges = np.arange(lim_min_bin, lim_max_bin + bin_width, bin_width)

    # 2D histogram
    hb = ax.hist2d(
        x,
        y,
        bins=[bins_edges, bins_edges],
        cmap="viridis",
        cmin=1,
        zorder=5,
        norm=mcolors.LogNorm(),
    )
    cbar = fig.colorbar(hb[3], ax=ax, shrink=0.7)
    cbar.set_label("Number of commits")
    cbar.solids.set_rasterized(False)

    # Highlight top-left: incremental > full
    x_fill = np.linspace(lim_min, lim_max, 500)
    ax.fill_betweenx(x_fill, lim_min, x_fill, color="red", alpha=0.2, zorder=0)

    # y=x reference
    ax.plot(
        [lim_min, lim_max],
        [lim_min, lim_max],
        "r--",
        linewidth=1,
        label="y=x",
        zorder=2,
    )

    # Labels back in ms scale
    ax.set_xlabel("Full Timings (ms)")
    ax.set_ylabel("Incremental Timings (ms)")

    from matplotlib.ticker import FuncFormatter, MultipleLocator, FixedLocator

    def log_formatter(x, pos):
        return f"$10^{{{x:.0f}}}$"

    ax.xaxis.set_major_formatter(FuncFormatter(log_formatter))
    ax.yaxis.set_major_formatter(FuncFormatter(log_formatter))

    # Use MultipleLocator for major ticks (every integer, which represents powers of 10)
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MultipleLocator(1))

    # For minor ticks, we need to manually place them at log positions
    # Between 10^n and 10^(n+1), place ticks at log10(2), log10(3), ..., log10(9)
    import math

    minor_ticks = []
    for i in range(int(np.floor(lim_min)), int(np.ceil(lim_max))):
        for j in range(2, 10):
            minor_ticks.append(i + math.log10(j))

    ax.xaxis.set_minor_locator(FixedLocator(minor_ticks))
    ax.yaxis.set_minor_locator(FixedLocator(minor_ticks))

    ax.grid(True, linestyle=":", linewidth=0.5, which="major")

    ax.set_xlim(lim_min, lim_max)
    ax.set_ylim(lim_min, lim_max)
    ax.set_aspect("equal", adjustable="box")

    save_plot(fig, output_file)


@analysis
def textual_analysis(df: pd.DataFrame, output_file: Path):
    results = "Notes:"
    results += "\n- Consider using a dependency graph to determine what to update."
    results += "\n\n" + "=" * 80 + "\n"

    times = df.groupby("same")[["full_timings_ms", "incremental_timings_ms"]].mean()
    results += "Timings by Same\n" + str(times)
    results += "\n\n" + "=" * 80 + "\n"

    df["speedup"] = df["full_timings_ms"] / df["incremental_timings_ms"]
    speedup = df.groupby("same")["speedup"].describe()
    results += "Speedup by Same\n" + str(speedup)
    results += "\n\n" + "=" * 80 + "\n"

    grouped = df.groupby(
        pd.cut(df["edits_count"], [0, 1, 3, 10, 30, 100, 500]), observed=True
    )["speedup"].median()
    results += "Speedup by Edits\n" + str(grouped)
    results += "\n\n" + "=" * 80 + "\n"

    bad_perf_small_edits = df.query("speedup < 1 and edits_count < 5")
    results += "speedup < 1 && edits < 5\n" + str(bad_perf_small_edits)
    results += "\n\n" + "=" * 80 + "\n"

    df["lines_per_edit"] = df["lines_changed"] / df["edits_count"]
    results += "Avg lines per edit\n" + str(df["lines_per_edit"].describe())
    results += "\n\n" + "=" * 80 + "\n"

    output_file.with_suffix(".txt").write_text(results)

    # Statistical comparison of full vs incremental timings
    print(f"\nStatistical comparison (Full vs Incremental):")

    # Use only rows where both timings are valid
    valid_mask = ~(
        np.isnan(df["full_timings_ms"]) | np.isnan(df["incremental_timings_ms"])
    )
    full_times = df.loc[valid_mask, "full_timings_ms"]
    incr_times = df.loc[valid_mask, "incremental_timings_ms"]

    if len(full_times) > 1:
        # Paired t-test (assumes normal distribution of differences)
        t_stat, t_pvalue = stats.ttest_rel(full_times, incr_times)
        print(f"  Paired t-test (May be biased due to Heteroscedasticity):")
        print(f"    t-statistic: {t_stat:.4f}, p-value: {t_pvalue:.4e}")

        # Wilcoxon signed-rank test (non-parametric, better for skewed data)
        w_stat, w_pvalue = stats.wilcoxon(full_times, incr_times)
        print(f"  Wilcoxon signed-rank test (Unbiased by Heteroscedasticity):")
        print(f"    W-statistic: {w_stat:.4f}, p-value: {w_pvalue:.4e}")

        # Effect size (Cohen's d)
        mean_diff = np.mean(full_times - incr_times)
        pooled_std = np.sqrt((np.var(full_times) + np.var(incr_times)) / 2)
        cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0
        print(f"  Effect size (Cohen's d): {cohens_d:.4f}")
        print(f"    Interpretation: ", end="")
        if abs(cohens_d) < 0.2:
            print("negligible")
        elif abs(cohens_d) < 0.5:
            print("small")
        elif abs(cohens_d) < 0.8:
            print("medium")
        else:
            print("large")

        # Speedup ratio analysis
        speedup = full_times / incr_times.replace(0, np.nan)
        speedup = speedup[~np.isnan(speedup)]
        print(f"  Speedup ratio (Full/Incremental):")
        print(f"    Mean: {np.mean(speedup):.4f}")
        print(f"    Median: {np.median(speedup):.4f}")
        print(f"    Std: {np.std(speedup):.4f}")
        print(
            f"    Q1-Q3: [{np.percentile(speedup, 25):.4f}, {np.percentile(speedup, 75):.4f}]"
        )
        faster_count = np.sum(speedup > 1)
        slower_count = np.sum(speedup < 1)
        print(
            f"    Full faster: {faster_count}/{len(speedup)} ({100*faster_count/len(speedup):.1f}%)"
        )
        print(
            f"    Incremental faster: {slower_count}/{len(speedup)} ({100*slower_count/len(speedup):.1f}%)"
        )


@analysis
def file_size_vs_timings(df: pd.DataFrame, output_file: Path):
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.set_size_inches(w=5.9, h=(5.9 / 10) * 6)

    sns.scatterplot(
        data=df,
        x="file_size_bytes",
        y="full_timings_ms",
        label="Full",
        alpha=0.7,
        ax=ax,
    )
    sns.scatterplot(
        data=df,
        x="file_size_bytes",
        y="incremental_timings_ms",
        label="Incremental",
        alpha=0.7,
        ax=ax,
    )

    # Compute log-transformed data for correlation and regression
    log_size = np.log10(df["file_size_bytes"].replace(0, np.nan))
    log_full = np.log10(df["full_timings_ms"].replace(0, np.nan))
    log_incr = np.log10(df["incremental_timings_ms"].replace(0, np.nan))

    # Remove NaN values for regression
    mask_full = ~(np.isnan(log_size) | np.isnan(log_full))
    mask_incr = ~(np.isnan(log_size) | np.isnan(log_incr))

    # Fit lines: log(y) = slope * log(x) + intercept
    slope_full = slope_incr = None
    if mask_full.sum() > 1:
        slope_full, intercept_full = np.polyfit(
            log_size[mask_full], log_full[mask_full], 1
        )

        # Compute residuals for heteroscedasticity-aware confidence interval
        log_x_full = log_size[mask_full]
        log_y_full = log_full[mask_full]
        y_pred_full = slope_full * log_x_full + intercept_full
        residuals_full = log_y_full - y_pred_full

        x_fit = np.logspace(
            np.log10(df["file_size_bytes"].min()),
            np.log10(df["file_size_bytes"].max()),
            100,
        )
        log_x_fit = np.log10(x_fit)
        y_fit_full = 10 ** (slope_full * log_x_fit + intercept_full)

        # Local weighted standard error using nearby residuals
        ci_full = []
        for x_val in log_x_fit:
            # Weight residuals by distance to this x value
            distances = np.abs(log_x_full - x_val)
            weights = np.exp(-distances / 0.5)  # Bandwidth of 0.5 in log space
            weights /= weights.sum()
            local_variance = np.sum(weights * residuals_full**2)
            local_se = np.sqrt(local_variance)
            ci_full.append(1.96 * local_se)

        ci_full = np.array(ci_full)
        y_upper_full = 10 ** (slope_full * log_x_fit + intercept_full + ci_full)
        y_lower_full = 10 ** (slope_full * log_x_fit + intercept_full - ci_full)

        ax.plot(
            x_fit,
            y_fit_full,
            "--",
            linewidth=2,
            label=f"Full fit ($\\propto$ size$^{{{slope_full:.2f}}}$)",
            zorder=10,
        )
        ax.fill_between(x_fit, y_lower_full, y_upper_full, alpha=0.2, zorder=5)

    if mask_incr.sum() > 1:
        slope_incr, intercept_incr = np.polyfit(
            log_size[mask_incr], log_incr[mask_incr], 1
        )

        # Compute residuals for heteroscedasticity-aware confidence interval
        log_x_incr = log_size[mask_incr]
        log_y_incr = log_incr[mask_incr]
        y_pred_incr = slope_incr * log_x_incr + intercept_incr
        residuals_incr = log_y_incr - y_pred_incr

        x_fit = np.logspace(
            np.log10(df["file_size_bytes"].min()),
            np.log10(df["file_size_bytes"].max()),
            100,
        )
        log_x_fit = np.log10(x_fit)
        y_fit_incr = 10 ** (slope_incr * log_x_fit + intercept_incr)

        # Local weighted standard error using nearby residuals to capture heteroscedasticity
        ci_incr = []
        for x_val in log_x_fit:
            # Weight residuals by distance to this x value
            distances = np.abs(log_x_incr - x_val)
            weights = np.exp(-distances / 0.5)  # Bandwidth of 0.5 in log space
            weights /= weights.sum()
            local_variance = np.sum(weights * residuals_incr**2)
            local_se = np.sqrt(local_variance)
            ci_incr.append(1.96 * local_se)

        ci_incr = np.array(ci_incr)
        y_upper_incr = 10 ** (slope_incr * log_x_fit + intercept_incr + ci_incr)
        y_lower_incr = 10 ** (slope_incr * log_x_fit + intercept_incr - ci_incr)

        ax.plot(
            x_fit,
            y_fit_incr,
            "--",
            linewidth=2,
            label=f"Incremental fit ($\\propto$ size$^{{{slope_incr:.2f}}}$)",
            zorder=10,
        )
        ax.fill_between(x_fit, y_lower_incr, y_upper_incr, alpha=0.2, zorder=5)

    ax.set_xscale("log")
    ax.set_yscale("log")

    # Set logarithmic tick locators and formatters
    ax.xaxis.set_major_locator(LogLocator(base=10.0, numticks=10))
    ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=10))
    ax.xaxis.set_minor_locator(
        LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=10)
    )
    ax.yaxis.set_minor_locator(
        LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=10)
    )

    ax.set_xlabel("File Size (bytes)")
    ax.set_ylabel("Time (ms)")

    ax.grid(True, linestyle=":", linewidth=0.5, which="major")
    ax.legend()
    save_plot(fig, output_file)

    corr_full = (
        np.corrcoef(log_size[mask_full], log_full[mask_full])[0, 1]
        if mask_full.sum() > 1
        else np.nan
    )
    corr_incr = (
        np.corrcoef(log_size[mask_incr], log_incr[mask_incr])[0, 1]
        if mask_incr.sum() > 1
        else np.nan
    )

    print(f"Correlation (log-log):")
    print(f"  File size ↔ Full timings:         {corr_full:.6f}")
    print(f"  File size ↔ Incremental timings:  {corr_incr:.6f}")
    if slope_full is not None:
        print(f"  Full timing scaling exponent:     {slope_full:.6f}")
    if slope_incr is not None:
        print(f"  Incremental timing scaling exponent: {slope_incr:.6f}")

    # Compute heteroscedasticity measures
    print(f"\nHeteroscedasticity analysis:")

    # For full timings
    if mask_full.sum() > 10:
        # Compute residuals
        log_x_full = log_size[mask_full]
        log_y_full = log_full[mask_full]
        y_pred_full = slope_full * log_x_full + intercept_full
        residuals_full = log_y_full - y_pred_full

        # Breusch-Pagan test: regress squared residuals on log_x
        bp_slope, bp_intercept = np.polyfit(log_x_full, residuals_full**2, 1)
        bp_pred = bp_slope * log_x_full + bp_intercept
        ss_total = np.sum((residuals_full**2 - np.mean(residuals_full**2)) ** 2)
        ss_resid = np.sum((residuals_full**2 - bp_pred) ** 2)
        r_squared = 1 - (ss_resid / ss_total) if ss_total > 0 else 0
        bp_statistic = len(log_x_full) * r_squared
        bp_pvalue = 1 - stats.chi2.cdf(bp_statistic, 1)

        # Variance ratio: upper 25% vs lower 25% of x values
        sorted_idx = np.argsort(log_x_full)
        n_quartile = len(log_x_full) // 4
        lower_var = np.var(residuals_full[sorted_idx[:n_quartile]])
        upper_var = np.var(residuals_full[sorted_idx[-n_quartile:]])
        var_ratio_full = upper_var / lower_var if lower_var > 0 else np.nan

        print(f"  Full timings:")
        print(f"    Breusch-Pagan statistic: {bp_statistic:.4f} (p={bp_pvalue:.4e})")
        print(f"    Variance ratio (Q4/Q1):  {var_ratio_full:.4f}")

    # For incremental timings
    if mask_incr.sum() > 10:
        # Compute residuals
        log_x_incr = log_size[mask_incr]
        log_y_incr = log_incr[mask_incr]
        y_pred_incr = slope_incr * log_x_incr + intercept_incr
        residuals_incr = log_y_incr - y_pred_incr

        # Breusch-Pagan test
        bp_slope, bp_intercept = np.polyfit(log_x_incr, residuals_incr**2, 1)
        bp_pred = bp_slope * log_x_incr + bp_intercept
        ss_total = np.sum((residuals_incr**2 - np.mean(residuals_incr**2)) ** 2)
        ss_resid = np.sum((residuals_incr**2 - bp_pred) ** 2)
        r_squared = 1 - (ss_resid / ss_total) if ss_total > 0 else 0
        bp_statistic = len(log_x_incr) * r_squared
        bp_pvalue = 1 - stats.chi2.cdf(bp_statistic, 1)

        # Variance ratio
        sorted_idx = np.argsort(log_x_incr)
        n_quartile = len(log_x_incr) // 4
        lower_var = np.var(residuals_incr[sorted_idx[:n_quartile]])
        upper_var = np.var(residuals_incr[sorted_idx[-n_quartile:]])
        var_ratio_incr = upper_var / lower_var if lower_var > 0 else np.nan

        print(f"  Incremental timings:")
        print(f"    Breusch-Pagan statistic: {bp_statistic:.4f} (p={bp_pvalue:.4e})")
        print(f"    Variance ratio (Q4/Q1):  {var_ratio_incr:.4f}")


@analysis
def edits_vs_timings_by_type(df: pd.DataFrame, output_file: Path):
    # Simple categorization: 0 = no edits, 1-5 = small, >5 = large
    df = df.copy()
    df["edit_size"] = pd.cut(
        df["lines_changed"],
        bins=[-1, 0, 5, 50, np.inf],
        labels=["None", "Small", "Medium", "Large"],
    )

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.set_size_inches(w=5.9, h=(5.9 / 12) * 6)
    sns.boxplot(x="edit_size", y="incremental_timings_ms", data=df, ax=ax)
    sns.boxplot(
        x="edit_size",
        y="full_timings_ms",
        data=df,
        ax=ax,
        boxprops={"facecolor": "none"},
        showcaps=False,
        showfliers=False,
    )

    ax.set_yscale("log")

    # Set logarithmic tick locators and formatters
    ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=10))
    ax.yaxis.set_minor_locator(
        LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=10)
    )

    ax.set_xlabel("Edit Size (LOC)")
    ax.set_ylabel("Time (ms)")

    save_plot(fig, output_file)


@analysis
def speedup_vs_lines_changed(df: pd.DataFrame, output_file: Path):
    df = df.copy()
    df["speedup"] = df["full_timings_ms"] / df["incremental_timings_ms"].replace(
        0, np.nan
    )

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.set_size_inches(w=5.9, h=(5.9 / 12) * 6)
    sns.scatterplot(x="lines_changed", y="speedup", alpha=0.7, data=df, ax=ax)
    ax.axhline(1, color="red", linestyle="--", label="No Speedup")
    ax.set_yscale("log")
    ax.set_xscale("log")

    # Set logarithmic tick locators and formatters
    ax.xaxis.set_major_locator(LogLocator(base=10.0, numticks=10))
    ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=10))
    ax.xaxis.set_minor_locator(
        LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=10)
    )
    ax.yaxis.set_minor_locator(
        LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=10)
    )

    ax.set_xlabel("Lines Changed (log)")
    ax.set_ylabel("Speedup (Full / Incremental)")

    ax.legend()
    save_plot(fig, output_file)


@analysis
def summary_table(df: pd.DataFrame, output_file: Path):
    df = df.copy()
    df["speedup"] = df["full_timings_ms"] / df["incremental_timings_ms"].replace(
        0, np.nan
    )

    # Compute summary statistics
    summary_general = {
        "Number of samples": len(df),
        "Fraction Faster Incremental": (df["speedup"] > 1).mean(),
        "Correlation Full vs Size": df[["file_size_bytes", "full_timings_ms"]]
        .corr()
        .iloc[0, 1],
        "Correlation Incr vs Size": df[["file_size_bytes", "incremental_timings_ms"]]
        .corr()
        .iloc[0, 1],
    }

    def describe(series):
        s = series.dropna()
        return {
            "P10": np.percentile(s, 10),
            "P25": np.percentile(s, 25),
            "P50": np.percentile(s, 50),
            "P75": np.percentile(s, 75),
            "P90": np.percentile(s, 90),
            "Mean": s.mean(),
            "Std. Dev": s.std(),
        }

    summary_metrics = {
        "Speedup (Full/Incremental)": describe(df["speedup"]),
        "Full Time (ms)": describe(df["full_timings_ms"]),
        "Incremental Time (ms)": describe(df["incremental_timings_ms"]),
        "File Size (bytes)": describe(df["file_size_bytes"]),
        "Edits Count": describe(df["edits_count"]),
    }

    # Build DataFrame
    summary_df = pd.DataFrame(summary_metrics).T[
        ["P10", "P25", "P50", "P75", "P90", "Mean", "Std. Dev"]
    ]

    # summary_df.index.name = "Metric"

    # Add general metrics as a header section
    general_df = pd.DataFrame(summary_general, index=["Value"]).T
    general_df.index.name = "General Metric"

    summary_df.rename(
        columns={
            col: f"\\boldmath\\textbf{{$P_{{{col[1:]}}}$}}\\unboldmath"
            for col in summary_df.columns
            if col.startswith("P")
        },
        inplace=True,
    )
    summary_df.rename(
        columns={
            col: f"\\textbf{{{col}}}"
            for col in summary_df.columns
            if col in ("Mean", "Std. Dev")
        },
        inplace=True,
    )

    # Export
    summary_df.to_csv(output_file.with_suffix(".csv"))
    summary_df.to_latex(
        output_file.with_suffix(".tex"),
        float_format=lambda x: f"{x:.2f}",
        column_format="@{}l" + "r" * len(summary_df.columns) + "@{}",
    )

    general_df.to_csv(output_file.with_name(output_file.stem + "_general.csv"))
    general_df.to_latex(
        output_file.with_name(output_file.stem + "_general.tex"), float_format="%.3f"
    )

    # Print summaries
    print("\n=== General Metrics ===")
    print(general_df.round(3))

    print("\n=== Distribution Summary ===")
    print(summary_df.round(3))


# --- Preprocessing --- #


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Numeric conversions
    for col in [
        "full_timings_ms",
        "incremental_timings_ms",
        "lines_changed",
        "file_size_bytes",
        "full_nodes",
        "full_edges",
        "incremental_nodes",
        "incremental_edges",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    df["full_timings_ms"] = df["full_timings_ms"].clip(lower=0.1)
    df["incremental_timings_ms"] = df["incremental_timings_ms"].clip(lower=0.1)

    df["patch_id"] = df["commit_hash"].str[:8]

    same_map = {0: "Different", 1: "Same", 2: "New", 3: "Error"}
    df["same"] = df["same"].apply(lambda x: same_map.get(x, "Unknown"))
    df = df[df["same"] != "New"]

    # File info
    df["directory"] = df["file_path"].str.split("/").str[0]
    df["extension"] = df["file_path"].str.split(".").str[-1]

    return df


# --- Main Logic --- #


def plot_metrics(df: pd.DataFrame, output_dir: Path):
    for analysis_func in ANALYSES:
        print(f"Running analysis: {analysis_func.__name__} ({len(df)} rows)")
        try:
            analysis_func(df.copy(), output_dir / analysis_func.__name__)
        except Exception as e:
            print(f"Error occurred while running {analysis_func.__name__}: {e}")


def main():
    base_dir = Path("repos/metrics")
    output_dir = base_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    all_dfs = []
    missing = 0
    for f in base_dir.glob("*.csv"):
        try:
            df = pd.read_csv(f)
            df["repo"] = f.name
            df = prepare_dataframe(df)
            all_dfs.append(df)
        except Exception:
            missing += 1

    print(f"Loaded metrics from {len(all_dfs)} repos, {missing} missing.")

    # Some simple textual metrics like the average times for each dir
    for df in all_dfs:
        print(f"Directory: {df['repo'].iloc[0]}")
        for col in df.columns:
            print(df[col].describe())

    if all_dfs:
        combined_df = pd.concat(all_dfs, ignore_index=True)
        plot_metrics(combined_df, output_dir)
        print(f"Metrics plots saved to {output_dir}")
    else:
        print("No metrics data found.")


if __name__ == "__main__":
    main()
