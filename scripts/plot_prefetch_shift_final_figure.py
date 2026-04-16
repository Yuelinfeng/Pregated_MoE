import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, Normalize
import pandas as pd
import seaborn as sns


CONDITION_ORDER = [
    "stable_homogeneous",
    "shifted_homogeneous",
    "stable_mixed",
    "shifted_mixed",
]

CONDITION_LABELS = {
    "stable_homogeneous": "Stable\nhomogeneous",
    "shifted_homogeneous": "Shifted\nhomogeneous",
    "stable_mixed": "Stable\nmixed",
    "shifted_mixed": "Shifted\nmixed",
}

TARGET_COHORT = "all"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--results_root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    return parser.parse_args()


def safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def load_latency(results_root: Path) -> pd.DataFrame:
    latency = pd.read_csv(results_root / "latency_compare.csv")
    latency["cache_ratio"] = latency["cache_ratio"].astype(float)
    latency["prefetch_over_on_demand"] = latency["prefetch_over_on_demand"].astype(float)
    latency["delta_ms"] = latency["prefetch_mean_ms"].astype(float) - latency["on_demand_mean_ms"].astype(float)
    latency["condition_label"] = latency["condition"].map(CONDITION_LABELS)
    return latency


def load_aggregated_f1(results_root: Path) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for csv_path in sorted((results_root / "prefetch").glob("*/cr*/confusion_boundary8.csv")):
        condition = csv_path.parts[-3]
        cache_ratio = float(csv_path.parts[-2].replace("cr", "", 1))
        frame = pd.read_csv(csv_path)
        subset = frame[frame["cohort"] == TARGET_COHORT]
        tp = int(subset["tp"].sum())
        fp = int(subset["fp"].sum())
        fn = int(subset["fn"].sum())
        tn = int(subset["tn"].sum())
        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        f1 = safe_div(2 * precision * recall, precision + recall) if (precision + recall) else 0.0
        rows.append(
            {
                "condition": condition,
                "condition_label": CONDITION_LABELS[condition],
                "cache_ratio": cache_ratio,
                "events": int(subset["events"].sum()),
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "f1_pct": f1 * 100.0,
                "avg_cache_hit_rate": float(subset["avg_cache_hit_rate"].mean()),
            }
        )
    return pd.DataFrame(rows)


def build_summary_table(latency: pd.DataFrame, f1_table: pd.DataFrame) -> pd.DataFrame:
    merged = latency.merge(
        f1_table[["condition", "cache_ratio", "events", "f1", "f1_pct", "avg_cache_hit_rate"]],
        on=["condition", "cache_ratio"],
        how="inner",
    )
    merged = merged.sort_values(["condition", "cache_ratio"]).reset_index(drop=True)
    return merged


def make_pivot(frame: pd.DataFrame, value_column: str) -> pd.DataFrame:
    pivot = frame.pivot(index="condition_label", columns="cache_ratio", values=value_column)
    ordered_index = [CONDITION_LABELS[name] for name in CONDITION_ORDER]
    pivot = pivot.reindex(index=ordered_index)
    pivot = pivot.reindex(columns=sorted(pivot.columns))
    return pivot


def make_annotation_table(frame: pd.DataFrame, kind: str) -> pd.DataFrame:
    ordered_index = [CONDITION_LABELS[name] for name in CONDITION_ORDER]
    ordered_columns = sorted(frame["cache_ratio"].unique())
    table = pd.DataFrame(index=ordered_index, columns=ordered_columns, dtype=object)

    for _, row in frame.iterrows():
        idx = row["condition_label"]
        col = row["cache_ratio"]
        if kind == "latency":
            table.loc[idx, col] = f"{row['prefetch_over_on_demand']:.3f}x\n{row['delta_ms']:+.1f} ms"
        else:
            table.loc[idx, col] = f"{row['f1_pct']:.3f}%\nN={row['events'] / 1000.0:.1f}k"
    return table


def draw_highlight_box(ax, row_idx: int, col_count: int, color: str = "#2b2b2b", lw: float = 2.2) -> None:
    rect = plt.Rectangle((0, row_idx), col_count, 1, fill=False, edgecolor=color, linewidth=lw)
    ax.add_patch(rect)


def annotate_heatmap(ax, value_pivot: pd.DataFrame, annot_table: pd.DataFrame, norm, cmap_name: str) -> None:
    cmap = plt.get_cmap(cmap_name)
    for row_idx, row_label in enumerate(value_pivot.index):
        for col_idx, col_label in enumerate(value_pivot.columns):
            value = value_pivot.loc[row_label, col_label]
            text = annot_table.loc[row_label, col_label]
            rgba = cmap(norm(value))
            luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            text_color = "white" if luminance < 0.55 else "#1f1f1f"
            ax.text(
                col_idx + 0.5,
                row_idx + 0.5,
                text,
                ha="center",
                va="center",
                color=text_color,
                fontsize=12,
                fontweight="medium",
            )


def plot_final_figure(summary: pd.DataFrame, output_dir: Path) -> Path:
    output_path = output_dir / "final_prefetch_shift_figure.png"
    output_dir.mkdir(parents=True, exist_ok=True)

    latency_pivot = make_pivot(summary, "prefetch_over_on_demand")
    f1_pivot = make_pivot(summary, "f1_pct")

    latency_annot = make_annotation_table(summary, "latency")
    f1_annot = make_annotation_table(summary, "f1")

    sns.set_theme(style="whitegrid", context="talk")
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 7.5), gridspec_kw={"width_ratios": [1, 1]})

    latency_norm = TwoSlopeNorm(vmin=summary["prefetch_over_on_demand"].min(), vcenter=1.0, vmax=summary["prefetch_over_on_demand"].max())
    f1_norm = Normalize(vmin=summary["f1_pct"].min(), vmax=summary["f1_pct"].max())

    heatmap_kws = dict(
        linewidths=1.2,
        linecolor="white",
        cbar_kws={"shrink": 0.88},
    )

    sns.heatmap(
        latency_pivot,
        cmap="RdBu_r",
        norm=latency_norm,
        ax=axes[0],
        **heatmap_kws,
    )
    annotate_heatmap(axes[0], latency_pivot, latency_annot, latency_norm, "RdBu_r")
    axes[0].set_title("System effect\nLatency ratio (prefetch / on-demand)")
    axes[0].set_xlabel("Cache ratio")
    axes[0].set_ylabel("")
    axes[0].collections[0].colorbar.set_label("Latency ratio")

    sns.heatmap(
        f1_pivot,
        cmap="RdYlBu",
        norm=f1_norm,
        ax=axes[1],
        **heatmap_kws,
    )
    annotate_heatmap(axes[1], f1_pivot, f1_annot, f1_norm, "RdYlBu")
    axes[1].set_title("Prediction evidence\nNext-layer expert-set F1")
    axes[1].set_xlabel("Cache ratio")
    axes[1].set_ylabel("")
    axes[1].collections[0].colorbar.set_label("F1 (%)")

    for ax in axes:
        ax.set_xticklabels([f"{tick:.2f}" if tick < 0.1 else f"{tick:.1f}" for tick in sorted(summary["cache_ratio"].unique())], rotation=0)
        ax.tick_params(axis="y", rotation=0)
        draw_highlight_box(ax, row_idx=3, col_count=len(latency_pivot.columns), color="#202020", lw=2.6)

    fig.suptitle(
        "Cross-request workload shift combined with intra-request domain mixing turns decoder prefetch from a benefit into a liability",
        y=0.98,
        fontsize=19,
    )
    fig.text(
        0.5,
        0.02,
        "Each cell on the right aggregates all decoder MoE transitions; annotation shows F1 and the number of evaluated transition events.",
        ha="center",
        va="center",
        fontsize=11,
    )

    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    fig.savefig(output_path, dpi=240)
    plt.close(fig)
    return output_path


def main() -> None:
    args = parse_args()
    latency = load_latency(args.results_root)
    f1_table = load_aggregated_f1(args.results_root)
    summary = build_summary_table(latency, f1_table)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_dir / "final_prefetch_shift_figure_data.csv", index=False)

    output_path = plot_final_figure(summary, args.output_dir)
    print(output_path)


if __name__ == "__main__":
    main()
