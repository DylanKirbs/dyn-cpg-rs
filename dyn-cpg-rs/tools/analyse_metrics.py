from typing import Callable, List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

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
    fig.savefig(output_file.with_suffix(".pdf"))
    # fig.savefig(output_file.with_suffix(".png"), dpi=150)
    plt.close(fig)


def use_logscale_if_wide(ax, column: pd.Series):
    finite_vals = column.replace([np.inf, -np.inf], np.nan).dropna()
    if not finite_vals.empty and finite_vals.max() / max(finite_vals.min(), 1e-9) > 10:
        ax.set_yscale("log")


# --- Analyses --- #


@analysis
def speedup_ratio(df: pd.DataFrame, output_file: Path):
    df["speedup"] = df["full_timings_ms"] / df["incremental_timings_ms"].replace(
        0, np.nan
    )

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(x="patch_type", y="speedup", data=df, ax=ax)
    ax.axhline(1, color="red", linestyle="--", label="No Speedup")
    ax.axhspan(0, 1, color="red", alpha=0.1)
    ax.set_ylabel("Full / Incremental")
    ax.set_title("Speedup Ratio by Patch Type")
    ax.legend()
    use_logscale_if_wide(ax, df["speedup"])
    save_plot(fig, output_file)


@analysis
def cumulative_timings(df: pd.DataFrame, output_file: Path):
    df_sorted = df.sort_values(["directory", "patch_index"])
    df_sorted["cumulative_full"] = df_sorted.groupby("directory")[
        "full_timings_ms"
    ].cumsum()
    df_sorted["cumulative_incremental"] = df_sorted.groupby("directory")[
        "incremental_timings_ms"
    ].cumsum()

    fig, ax = plt.subplots(figsize=(10, 6))
    for directory, sub in df_sorted.groupby("directory"):
        ax.plot(sub["patch_index"], sub["cumulative_full"], label=f"{directory} full")
        ax.plot(
            sub["patch_index"],
            sub["cumulative_incremental"],
            label=f"{directory} inc",
            linestyle="--",
        )

    ax.set_title("Cumulative Timings over Patch Sequence")
    ax.set_xlabel("Patch Index")
    ax.set_ylabel("Cumulative Time (ms)")
    use_logscale_if_wide(
        ax,
        pd.concat([df_sorted["cumulative_full"], df_sorted["cumulative_incremental"]]),
    )
    ax.legend()
    save_plot(fig, output_file)


@analysis
def directory_means(df: pd.DataFrame, output_file: Path):
    agg = (
        df.groupby("directory")[["full_timings_ms", "incremental_timings_ms"]]
        .mean()
        .reset_index()
    )
    agg = agg.melt(id_vars="directory", var_name="timing_type", value_name="ms")

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x="directory", y="ms", hue="timing_type", data=agg, ax=ax)
    ax.set_title("Average Timings per Directory")
    ax.set_ylabel("Time (ms)")
    use_logscale_if_wide(ax, agg["ms"])
    save_plot(fig, output_file)


@analysis
def edits_vs_timing(df: pd.DataFrame, output_file: Path):
    fig, ax = plt.subplots(figsize=(10, 6))

    # Define markers based on same value
    markers = {0: "x", 1: "o"}

    # Plot incremental timings
    for same_val in [0, 1]:
        mask = df["same"] == same_val
        if mask.any():
            sns.scatterplot(
                x="edits_count",
                y="incremental_timings_ms",
                data=df[mask],
                marker=markers[same_val],
                label=f"Incremental (same={same_val})",
                alpha=0.7,
                ax=ax,
            )

    # Plot full timings
    sns.scatterplot(
        x="edits_count",
        y="full_timings_ms",
        data=df,
        marker="o",
        label=f"Full",
        alpha=0.7,
        ax=ax,
    )

    ax.set_title("Timings vs Edits Count")
    ax.set_xlabel("Edits Count")
    ax.set_ylabel("Time (ms)")
    use_logscale_if_wide(
        ax, pd.concat([df["full_timings_ms"], df["incremental_timings_ms"]])
    )
    ax.legend()
    save_plot(fig, output_file)


@analysis
def patch_type_timings(df: pd.DataFrame, output_file: Path):
    order = sorted(df["patch_type"].dropna().unique())
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    for ax, col, title in zip(
        axes, ["full_timings_ms", "incremental_timings_ms"], ["Full", "Incremental"]
    ):
        sns.violinplot(x="patch_type", y=col, data=df, ax=ax, order=order, inner=None)
        sns.boxplot(
            x="patch_type",
            y=col,
            data=df,
            ax=ax,
            order=order,
            showcaps=False,
            boxprops={"facecolor": "none"},
            showfliers=False,
            whiskerprops={"linewidth": 0},
        )
        ax.set_title(f"{title} Timings by Patch Type")
        ax.set_xlabel("Patch Type")
        ax.set_ylabel("Time (ms)")

    save_plot(fig, output_file)


@analysis
def incremental_vs_full(df: pd.DataFrame, output_file: Path):
    fig, ax = plt.subplots(figsize=(8, 8))
    sns.scatterplot(
        x="full_timings_ms",
        y="incremental_timings_ms",
        hue="edits_count",
        data=df,
        alpha=0.7,
        ax=ax,
    )

    lims = [
        max(min(df["full_timings_ms"].min(), df["incremental_timings_ms"].min()), 1),
        max(df["full_timings_ms"].max(), df["incremental_timings_ms"].max()) * 1.1,
    ]
    ax.plot(lims, lims, "r--", linewidth=1, label="y=x")
    ax.fill_between(lims, lims, lims[1], color="red", alpha=0.1)
    ax.set_xlim(lims)  # type: ignore
    ax.set_ylim(lims)  # type: ignore

    ax.set_title("Incremental vs Full Timings")
    ax.set_xlabel("Full Timings (ms)")
    ax.set_ylabel("Incremental Timings (ms)")
    ax.set_aspect("equal", "box")
    ax.grid(True, linestyle=":", linewidth=0.5)
    ax.legend(title="Edits Count")
    ax.set_xscale("log")
    ax.set_yscale("log")

    save_plot(fig, output_file)


@analysis
def prop_same_by_patch_type(df: pd.DataFrame, output_file: Path):
    # aggregate counts per patch_type x same
    agg = (
        df.groupby(["patch_type", "same"])["edits_count"]
        .count()
        .reset_index()
        .rename(columns={"edits_count": "count"})
    )

    # map 0/1 to labels and pivot to get counts in separate columns
    agg["same"] = agg["same"].map({1: "Same", 0: "Different"})
    pivot = agg.pivot(index="patch_type", columns="same", values="count").fillna(0)

    # ensure both columns exist in a stable order
    for col in ("Different", "Same"):
        if col not in pivot.columns:
            pivot[col] = 0
    pivot = pivot[["Same", "Different"]]

    # convert counts to proportions per patch_type
    proportions = pivot.div(pivot.sum(axis=1), axis=0)

    # plot stacked bar of proportions
    fig, ax = plt.subplots(figsize=(12, 6))
    proportions.plot(kind="bar", stacked=True, ax=ax)
    ax.set_title("Proportion of Same vs Different CPGs by Patch Type")
    ax.set_ylabel("Proportion")
    ax.set_xlabel("Patch Type")
    ax.set_ylim(0, 1)

    import matplotlib.ticker as mtick

    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

    ax.legend()
    save_plot(fig, output_file)


# --- Preprocessing --- #


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    # edits_count full_timings_ms incremental_timings_ms same patch_index patch_name patch_type directory
    df = df.copy()

    # Make numeric and bump the min to 0.1ms to avoid log(0) issues
    df["full_timings_ms"] = pd.to_numeric(df["full_timings_ms"], errors="coerce").clip(
        lower=0.1
    )
    df["incremental_timings_ms"] = pd.to_numeric(
        df["incremental_timings_ms"], errors="coerce"
    ).clip(lower=0.1)

    df["patch_type"] = df["patch_name"].str.split("_", n=2).str[1]
    df["patch_index"] = pd.to_numeric(
        df["patch_name"].str.split("_", n=1).str[0], errors="coerce"
    )
    df = df.dropna(subset=["patch_index"])
    df["patch_index"] = df["patch_index"].astype(int)
    return df.sort_values(["patch_index", "patch_type"])


# --- Main Logic --- #


def plot_metrics(df: pd.DataFrame, output_dir: Path):
    df = prepare_dataframe(df)
    for analysis_func in ANALYSES:
        print(f"Running analysis: {analysis_func.__name__} ({len(df)} rows)")
        try:
            analysis_func(df.copy(), output_dir / analysis_func.__name__)
        except Exception as e:
            print(f"Error occurred while running {analysis_func.__name__}: {e}")


def load_metrics_from_directory(directory: Path) -> pd.DataFrame:
    metrics_file = directory / "metrics.csv"
    if not metrics_file.exists():
        raise FileNotFoundError(f"No metrics.csv found in {directory}")
    return pd.read_csv(metrics_file)


def main():
    base_dir = Path("seq_patches")
    output_dir = Path("seq_patches/metrics")
    output_dir.mkdir(parents=True, exist_ok=True)

    all_dfs = []
    missing = 0
    for subdir in base_dir.iterdir():
        if subdir.is_dir():
            try:
                df = load_metrics_from_directory(subdir)
                df["directory"] = subdir.name
                all_dfs.append(df)
            except FileNotFoundError:
                missing += 1

    print(f"Loaded metrics from {len(all_dfs)} directories, {missing} missing.")

    # Some simple textual metrics like the average times for each dir
    for df in all_dfs:
        print(f"Directory: {df['directory'].iloc[0]}\n{df.describe()}")

    if all_dfs:
        combined_df = pd.concat(all_dfs, ignore_index=True)
        plot_metrics(combined_df, output_dir)
        print(f"Metrics plots saved to {output_dir}")
    else:
        print("No metrics data found.")


if __name__ == "__main__":
    main()
