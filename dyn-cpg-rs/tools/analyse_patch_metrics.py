#!/usr/bin/env python3

from cProfile import label
from typing import Callable, List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from matplotlib.ticker import LogLocator

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

# --- Stellenbosch University Colors ---

# Primary colors
SUN_CONFIDENT_MAROON = "#61223b"
SUN_BRILLIANT_GOLD = "#b79962"

# Secondary colors
SUN_FIERY_ORANGE = "#dc4405"
SUN_WINE_RED = "#a60a3d"
SUN_SOIL = "#643335"
SUN_SEABREEZE_GREEN = "#82ccae"

# Faculty colors
SUN_AGRISCIENCE_GREEN = "#658d1b"
SUN_ARTS_ORANGE = "#ff9e1b"
SUN_EMS_TURQUOISE = "#2cccd3"
SUN_EDUCATION_BLUE = "#223d71"
SUN_ENGINEERING_YELLOW = "#eba900"
SUN_LAW_RED = "#bf0d3e"
SUN_MHS_TEAL = "#005f61"
SUN_MIL_SCIENCE_PEACH = "#e56a54"
SUN_SCIENCE_RED = "#d22730"
SUN_THEOLOGY_PURPLE = "#5c068c"

# Ancillary colors
SUN_GREY = "#7e7d7c"
SUN_GREY_DARK = "#212121"
SUN_GREY_LIGHT = "#fafafa"

# Create a custom color palette
SUN_PALETTE = [
    SUN_CONFIDENT_MAROON,
    SUN_BRILLIANT_GOLD,
    SUN_FIERY_ORANGE,
    SUN_SEABREEZE_GREEN,
    SUN_EDUCATION_BLUE,
    SUN_MHS_TEAL,
]

# --- Config --- #

sns.set_palette(SUN_PALETTE)
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
def speedup_ratio(df: pd.DataFrame, output_file: Path):
    df["speedup"] = df["full_timings_ms"] / df["incremental_timings_ms"].replace(
        0, np.nan
    )

    order = sorted(df["patch_type"].dropna().unique())

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.set_size_inches(w=5.9, h=(5.9 / 12) * 6)
    sns.boxplot(x="patch_type", y="speedup", data=df, ax=ax, order=order)
    ax.axhline(1, color=SUN_SCIENCE_RED, linestyle="--", label="No Speedup")
    ax.axhspan(0, 1, color=SUN_SCIENCE_RED, alpha=0.1)

    ax.set_xlabel("Patch Type")
    ax.set_ylabel("Full / Incremental")

    ax.legend()
    ax.set_yscale("log")
    ax.set_ylim(1 / 25.0, 25.0)
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
    fig.set_size_inches(w=5.9, h=(5.9 / 10) * 6)
    for directory, sub in df_sorted.groupby("directory"):
        ax.plot(sub["patch_index"], sub["cumulative_full"], label=f"{directory} full")
        ax.plot(
            sub["patch_index"],
            sub["cumulative_incremental"],
            label=f"{directory} inc",
            linestyle="--",
        )

    ax.set_xlabel("Patch Index")
    ax.set_ylabel("Cumulative Time (ms)")
    ax.set_yscale("log")
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
    fig.set_size_inches(w=5.9, h=(5.9 / 10) * 6)
    sns.barplot(x="directory", y="ms", hue="timing_type", data=agg, ax=ax)

    ax.set_ylabel("Time (ms)")
    ax.set_xlabel("Directory")
    ax.set_yscale("log")
    save_plot(fig, output_file)


@analysis
def edits_vs_timing(df: pd.DataFrame, output_file: Path):
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.set_size_inches(w=5.9, h=(5.9 / 10) * 6)

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

    ax.set_xlabel("Edits Count")
    ax.set_ylabel("Time (ms)")
    ax.set_xscale("log")
    ax.legend()
    save_plot(fig, output_file)


@analysis
def patch_type_timings(df: pd.DataFrame, output_file: Path):
    order = sorted(df["patch_type"].dropna().unique())

    # remove Refactor from order due to extreme outliers
    if "Refactor" in order:
        order.remove("Refactor")

    # Prepare data in long format for grouped plotting
    df_filtered = df[df["patch_type"].isin(order)].copy()
    df_long = df_filtered.melt(
        id_vars=["patch_type"],
        value_vars=["full_timings_ms", "incremental_timings_ms"],
        var_name="timing_type",
        value_name="time_ms",
    )
    df_long["timing_type"] = df_long["timing_type"].map(
        {"full_timings_ms": "Full", "incremental_timings_ms": "Incremental"}
    )

    fig, ax = plt.subplots(figsize=(9, 6))
    fig.set_size_inches(w=5.9, h=(5.9 / 9) * 6)

    # Create grouped box plot with different colors
    sns.boxplot(
        x="patch_type",
        y="time_ms",
        hue="timing_type",
        data=df_long,
        ax=ax,
        order=order,
    )

    ax.set_xlabel("Patch Type")
    ax.set_ylabel("Time (ms)")

    # Disable legend
    ax.legend().remove()

    save_plot(fig, output_file)


@analysis
def rename_local_timings(df: pd.DataFrame, output_file: Path):
    df_rl = df[df["patch_type"] == "Refactor"].copy()

    if df_rl.empty:
        print("No Refactor data found, skipping rename_local_timings")
        return

    # Prepare data in long format for grouped plotting
    df_long = df_rl.melt(
        id_vars=["patch_type"],
        value_vars=["full_timings_ms", "incremental_timings_ms"],
        var_name="timing_type",
        value_name="time_ms",
    )
    df_long["timing_type"] = df_long["timing_type"].map(
        {"full_timings_ms": "Full", "incremental_timings_ms": "Incremental"}
    )

    fig, ax = plt.subplots(figsize=(4, 4))
    fig.set_size_inches(w=5.9, h=(5.9 / 4) * 4)

    # Create grouped box plot with different colors
    sns.boxplot(
        x="patch_type",
        y="time_ms",
        hue="timing_type",
        data=df_long,
        ax=ax,
    )

    ax.set_xlabel("Refactor Patch Type")
    ax.set_ylabel("Time (ms)")

    # Disable legend
    ax.legend().remove()

    save_plot(fig, output_file)


@analysis
def incremental_vs_full(df: pd.DataFrame, output_file: Path):
    fig, ax = plt.subplots(figsize=(8, 8))
    fig.set_size_inches(w=5.9 / 2, h=5.9 / 2)
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
    ax.plot(lims, lims, "--", color=SUN_CONFIDENT_MAROON, linewidth=1, label="y=x")
    ax.fill_between(lims, lims, lims[1], color=SUN_SCIENCE_RED, alpha=0.1)
    ax.set_xlim(lims)  # type: ignore
    ax.set_ylim(lims)  # type: ignore

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
    fig.set_size_inches(w=5.9, h=(5.9 / 12) * 6)
    proportions.plot(kind="bar", stacked=True, ax=ax)

    ax.set_ylabel("Proportion")
    ax.set_xlabel("Patch Type")
    ax.set_ylim(0, 1)

    import matplotlib.ticker as mtick

    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

    ax.legend()
    save_plot(fig, output_file)


# --- Preprocessing --- #


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    # patch_name,edits_count,full_timings_ms,incremental_timings_ms,same,full_nodes,full_edges,incremental_nodes,incremental_edges,lines_changed
    df = df.copy()

    # Make numeric and bump the min to 0.1ms to avoid log(0) issues
    df["full_timings_ms"] = pd.to_numeric(df["full_timings_ms"], errors="coerce").clip(
        lower=0.1
    )
    df["incremental_timings_ms"] = pd.to_numeric(
        df["incremental_timings_ms"], errors="coerce"
    ).clip(lower=0.1)

    df["patch_type"] = df["patch_name"].str.split("_", n=2).str[1]

    pt_map = {
        "add-comment": "Insert",
        "insert-decl": "Insert",
        "delete-stmt": "Delete",
        "modify-num-lit": "Modify",
        "rename-local": "Refactor",
    }
    df["patch_type"] = df["patch_type"].map(pt_map).fillna(df["patch_type"])

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
