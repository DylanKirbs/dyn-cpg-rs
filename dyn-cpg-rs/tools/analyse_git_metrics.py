#!/usr/bin/env python3

from typing import Callable, List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import matplotlib.colors as mcolors

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
    plt.close(fig)


# --- Analyses --- #

@analysis
def incremental_vs_full_by_edits(df: pd.DataFrame, output_file: Path, hue_cap=25):

    df = df.copy()
    norm = mcolors.Normalize(vmin=df["edits_count"].min(), vmax=hue_cap)
    cmap = plt.get_cmap("viridis")
    colors = cmap(norm(df["edits_count"].to_numpy()))

    fig, ax = plt.subplots(figsize=(8, 8))
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

    ax.set_title("Incremental vs Full Timings (Log-Log) by Edits Count")
    ax.set_xlabel("Full Timings (ms, log)")
    ax.set_ylabel("Incremental Timings (ms, log)")
    ax.grid(True, linestyle=":", linewidth=0.5)
    ax.legend()

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label="Edits Count (capped at {})".format(hue_cap))

    save_plot(fig, output_file)


@analysis
def incremental_vs_full_heatmap(df: pd.DataFrame, output_file: Path):
    fig, ax = plt.subplots(figsize=(8, 8))

    # Log-transform
    x = np.log10(df["full_timings_ms"].clip(lower=0.1))
    y = np.log10(df["incremental_timings_ms"].clip(lower=0.1))

    # Limits
    lim_min = min(x.min(), y.min()) * 1.1
    lim_max = max(x.max(), y.max()) * 1.1

    # Highlight top-left: incremental > full
    x_fill = np.linspace(lim_min, lim_max, 500)
    ax.fill_betweenx(x_fill, lim_min, x_fill, color="red", alpha=0.2, zorder=0)

    # y=x reference
    ax.plot([lim_min, lim_max], [lim_min, lim_max], "r--", linewidth=1, label="y=x")

    # Bins
    bin_width = 0.2
    lim_min_bin = np.floor(lim_min / bin_width) * bin_width
    lim_max_bin = np.ceil(lim_max / bin_width) * bin_width
    bins_edges = np.arange(lim_min_bin, lim_max_bin + bin_width, bin_width)

    # 2D histogram
    hb = ax.hist2d(x, y, bins=[bins_edges, bins_edges], cmap="viridis", cmin=1)
    cbar = fig.colorbar(hb[3], ax=ax)
    cbar.set_label("Number of patches")


    # Labels back in ms scale
    ax.set_xlabel("Full Timings (ms, log10)")
    ax.set_ylabel("Incremental Timings (ms, log10)")
    ax.set_title("Heatmap of Incremental vs Full Timings")
    ax.grid(True, linestyle=":", linewidth=0.5)

    ax.set_xlim(lim_min, lim_max)
    ax.set_ylim(lim_min, lim_max)

    save_plot(fig, output_file)


@analysis
def speedup_ratio(df: pd.DataFrame, output_file: Path):
    df = df.copy()[df["edits_count"] < 25]
    df["speedup"] = df["full_timings_ms"] / df["incremental_timings_ms"].replace(
        0, np.nan
    ).replace(0.1, np.nan)

    fig, ax = plt.subplots(figsize=(12, 6))
    # Violin first (distribution)
    sns.violinplot(
        x="edits_count",
        y="speedup",
        data=df,
        ax=ax,
        inner=None,
        cut=0
    )
    # Box on top (central tendency)
    sns.boxplot(
        x="edits_count",
        y="speedup",
        data=df,
        ax=ax,
        showcaps=False,
        boxprops={"facecolor": "none"},
        showfliers=False,
        whiskerprops={"linewidth": 0},
    )

    ax.axhline(1, color="red", linestyle="--", label="No Speedup")
    ax.axhspan(0, 1, color="red", alpha=0.1)

    ax.set_ylabel("Full / Incremental")
    ax.set_title("Speedup Ratio by Number of edits (capped at 25)")
    ax.legend()
    ax.set_yscale("log")
    ax.set_ylim(1 / 50.0, 50.0)

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

    grouped = df.groupby(pd.cut(df["edits_count"], [0,1,3,10,30,100,500]), observed=True)["speedup"].median()
    results += "Speedup by Edits\n" + str(grouped)
    results += "\n\n" + "=" * 80 + "\n"

    bad_perf_small_edits = df.query("speedup < 1 and edits_count < 5")
    results += "speedup < 1 && edits < 5\n" + str(bad_perf_small_edits)
    results += "\n\n" + "=" * 80 + "\n"

    df["lines_per_edit"] = df["lines_changed"] / df["edits_count"]
    results += "Avg lines per edit\n" + str(df["lines_per_edit"].describe())
    results += "\n\n" + "=" * 80 + "\n"


    output_file.with_suffix(".txt").write_text(results)



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

    same_map = {
        0: "Different",
        1: "Same",
        2: "New",
        3: "Error"
    }
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
