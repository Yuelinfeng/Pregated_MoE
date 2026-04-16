import argparse
import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


CONDITION_ORDER = [
    "stable_homogeneous",
    "shifted_homogeneous",
    "stable_mixed",
    "shifted_mixed",
]

CONDITION_LABELS = {
    "stable_homogeneous": "Stable + homogeneous",
    "shifted_homogeneous": "Shifted + homogeneous",
    "stable_mixed": "Stable + mixed",
    "shifted_mixed": "Shifted + mixed",
}

CONDITION_COLORS = {
    "stable_homogeneous": "#2f5d8a",
    "shifted_homogeneous": "#77a6d1",
    "stable_mixed": "#c46a2d",
    "shifted_mixed": "#d94841",
}

TARGET_COHORTS = {"all", "boundary_window_8", "shift_boundary"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--results_root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--layerwise_cache_ratios", nargs="*", default=["0.03", "0.4"])
    parser.add_argument("--domain_order_cache_ratio", type=str, default="0.03")
    return parser.parse_args()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def load_latency(results_root: Path) -> pd.DataFrame:
    latency_path = results_root / "latency_compare.csv"
    latency = pd.read_csv(latency_path)
    latency["cache_ratio"] = latency["cache_ratio"].astype(float)
    latency["prefetch_over_on_demand"] = latency["prefetch_over_on_demand"].astype(float)
    latency["delta_ms"] = latency["prefetch_mean_ms"].astype(float) - latency["on_demand_mean_ms"].astype(float)
    latency["condition_label"] = latency["condition"].map(CONDITION_LABELS)
    return latency


def iter_confusion_files(results_root: Path) -> Iterable[Tuple[str, str, Path]]:
    prefetch_root = results_root / "prefetch"
    for csv_path in sorted(prefetch_root.glob("*/cr*/confusion_boundary8.csv")):
        condition = csv_path.parts[-3]
        cache_ratio = csv_path.parts[-2].replace("cr", "", 1)
        yield condition, cache_ratio, csv_path


def aggregate_confusion(results_root: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    aggregated_rows: List[Dict[str, object]] = []
    layerwise_rows: List[Dict[str, object]] = []

    for condition, cache_ratio, csv_path in iter_confusion_files(results_root):
        frame = pd.read_csv(csv_path)
        frame["condition"] = condition
        frame["cache_ratio"] = float(cache_ratio)
        frame["condition_label"] = CONDITION_LABELS[condition]

        for _, row in frame.iterrows():
            if row["cohort"] not in TARGET_COHORTS and not str(row["cohort"]).startswith("domain:"):
                continue
            layerwise_rows.append(
                {
                    "condition": condition,
                    "condition_label": CONDITION_LABELS[condition],
                    "cache_ratio": float(cache_ratio),
                    "cohort": row["cohort"],
                    "source_layer": row["source_layer"],
                    "target_layer": row["target_layer"],
                    "transition": shorten_transition(row["source_layer"], row["target_layer"]),
                    "events": int(row["events"]),
                    "tp": int(row["tp"]),
                    "fp": int(row["fp"]),
                    "fn": int(row["fn"]),
                    "tn": int(row["tn"]),
                    "precision": float(row["precision"]),
                    "recall": float(row["recall"]),
                    "f1": float(row["f1"]),
                    "specificity": float(row["specificity"]),
                    "balanced_accuracy": float(row["balanced_accuracy"]),
                    "weighted_recall": float(row["weighted_recall"]),
                    "avg_cache_hit_rate": float(row["avg_cache_hit_rate"]),
                    "avg_max_active_experts": float(row["avg_max_active_experts"]),
                    "avg_active_experts": float(row["avg_active_experts"]),
                }
            )

        grouped = frame[frame["cohort"].isin(TARGET_COHORTS)].groupby("cohort")
        for cohort, group in grouped:
            tp = int(group["tp"].sum())
            fp = int(group["fp"].sum())
            fn = int(group["fn"].sum())
            tn = int(group["tn"].sum())
            precision = safe_div(tp, tp + fp)
            recall = safe_div(tp, tp + fn)
            f1 = safe_div(2 * precision * recall, precision + recall) if (precision + recall) else 0.0
            specificity = safe_div(tn, tn + fp)
            balanced_accuracy = 0.5 * (recall + specificity)

            aggregated_rows.append(
                {
                    "condition": condition,
                    "condition_label": CONDITION_LABELS[condition],
                    "cache_ratio": float(cache_ratio),
                    "cohort": cohort,
                    "events": int(group["events"].sum()),
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "tn": tn,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "specificity": specificity,
                    "balanced_accuracy": balanced_accuracy,
                    "avg_cache_hit_rate": float(group["avg_cache_hit_rate"].mean()),
                    "avg_max_active_experts": float(group["avg_max_active_experts"].mean()),
                    "avg_active_experts": float(group["avg_active_experts"].mean()),
                }
            )

    aggregated = pd.DataFrame(aggregated_rows)
    layerwise = pd.DataFrame(layerwise_rows)
    return aggregated, layerwise


def shorten_transition(source_layer: str, target_layer: str) -> str:
    source_idx = source_layer.split("layer")[-1]
    target_idx = target_layer.split("layer")[-1]
    return f"L{source_idx}->L{target_idx}"


def safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def save_table(frame: pd.DataFrame, path: Path) -> None:
    ensure_parent(path)
    frame.to_csv(path, index=False)


def plot_latency_ratio(latency: pd.DataFrame, output_dir: Path) -> Path:
    figure_path = output_dir / "figure1_latency_ratio_vs_cache.png"
    ensure_parent(figure_path)

    sns.set_theme(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=(10, 6))

    for condition in CONDITION_ORDER:
        subset = latency[latency["condition"] == condition].sort_values("cache_ratio")
        ax.plot(
            subset["cache_ratio"],
            subset["prefetch_over_on_demand"],
            marker="o",
            linewidth=2.5,
            markersize=8,
            color=CONDITION_COLORS[condition],
            label=CONDITION_LABELS[condition],
        )

    ax.axhline(1.0, linestyle="--", linewidth=1.5, color="#444444", alpha=0.8)
    ax.set_xlabel("Cache ratio")
    ax.set_ylabel("Latency ratio (prefetch / on-demand)")
    ax.set_title("System outcome: when does prefetch still help?")
    ax.set_xticks(sorted(latency["cache_ratio"].unique()))
    ax.legend(frameon=True, loc="best")
    ax.text(
        0.402,
        1.002,
        "1.0 = break-even",
        fontsize=10,
        color="#444444",
        ha="left",
        va="bottom",
    )
    fig.tight_layout()
    fig.savefig(figure_path, dpi=220)
    plt.close(fig)
    return figure_path


def plot_f1_vs_cache(aggregated: pd.DataFrame, output_dir: Path) -> Path:
    figure_path = output_dir / "figure2_prediction_f1_vs_cache.png"
    ensure_parent(figure_path)

    all_rows = aggregated[aggregated["cohort"] == "all"].copy()
    all_rows["f1_pct"] = all_rows["f1"] * 100.0

    sns.set_theme(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=(10, 6))

    for condition in CONDITION_ORDER:
        subset = all_rows[all_rows["condition"] == condition].sort_values("cache_ratio")
        ax.plot(
            subset["cache_ratio"],
            subset["f1_pct"],
            marker="o",
            linewidth=2.5,
            markersize=8,
            color=CONDITION_COLORS[condition],
            label=CONDITION_LABELS[condition],
        )

    ax.set_xlabel("Cache ratio")
    ax.set_ylabel("Expert-set prediction F1 (%)")
    ax.set_title("Prediction quality: next-layer expert overlap under workload shift")
    ax.set_xticks(sorted(all_rows["cache_ratio"].unique()))
    ax.legend(frameon=True, loc="best")
    fig.tight_layout()
    fig.savefig(figure_path, dpi=220)
    plt.close(fig)
    return figure_path


def plot_layerwise_mixed(layerwise: pd.DataFrame, cache_ratios: List[str], output_dir: Path) -> Path:
    figure_path = output_dir / "figure3_mixed_layerwise_f1.png"
    ensure_parent(figure_path)

    chosen = [float(value) for value in cache_ratios]
    subset = layerwise[
        (layerwise["cohort"] == "all")
        & (layerwise["condition"].isin(["stable_mixed", "shifted_mixed"]))
        & (layerwise["cache_ratio"].isin(chosen))
    ].copy()
    subset["f1_pct"] = subset["f1"] * 100.0

    transitions = ["L1->L3", "L3->L5", "L5->L7", "L7->L9", "L9->L11"]
    available_ratios = [ratio for ratio in chosen if ratio in subset["cache_ratio"].unique()]
    if not available_ratios:
        raise RuntimeError("No layerwise mixed rows found for the requested cache ratios.")

    sns.set_theme(style="whitegrid", context="talk")
    fig, axes = plt.subplots(1, len(available_ratios), figsize=(6 * len(available_ratios), 5.5), sharey=True)
    if len(available_ratios) == 1:
        axes = [axes]

    for ax, cache_ratio in zip(axes, available_ratios):
        cache_rows = subset[subset["cache_ratio"] == cache_ratio]
        for condition in ["stable_mixed", "shifted_mixed"]:
            line = cache_rows[cache_rows["condition"] == condition].copy()
            line["transition"] = pd.Categorical(line["transition"], categories=transitions, ordered=True)
            line = line.sort_values("transition")
            ax.plot(
                line["transition"],
                line["f1_pct"],
                marker="o",
                linewidth=2.5,
                markersize=8,
                color=CONDITION_COLORS[condition],
                label=CONDITION_LABELS[condition],
            )

        ax.set_title(f"Cache ratio = {cache_ratio:g}")
        ax.set_xlabel("Decoder MoE transition")
        ax.tick_params(axis="x", rotation=20)

    axes[0].set_ylabel("Expert-set prediction F1 (%)")
    axes[-1].legend(frameon=True, loc="best")
    fig.suptitle("Where the mixed workload degrades next-layer prefetch prediction", y=1.02)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return figure_path


def plot_shifted_mixed_domain_order(layerwise: pd.DataFrame, cache_ratio: str, output_dir: Path) -> Path:
    figure_path = output_dir / "figure4_shifted_mixed_domain_order.png"
    ensure_parent(figure_path)

    ratio = float(cache_ratio)
    subset = layerwise[
        (layerwise["condition"] == "shifted_mixed")
        & (layerwise["cache_ratio"] == ratio)
        & (layerwise["cohort"].str.startswith("domain:"))
    ].copy()
    if subset.empty:
        raise RuntimeError("No shifted_mixed domain-order rows found for the requested cache ratio.")

    subset["f1_pct"] = subset["f1"] * 100.0
    subset["domain_order"] = subset["cohort"].str.replace("domain:", "", regex=False)
    subset["domain_order"] = subset["domain_order"].map(
        {
            "mixed:translation+summarization": "translation -> summarization",
            "mixed:summarization+translation": "summarization -> translation",
        }
    )
    transitions = ["L1->L3", "L3->L5", "L5->L7", "L7->L9", "L9->L11"]
    subset = subset.sort_values(["transition", "domain_order"])

    sns.set_theme(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        data=subset,
        x="transition",
        y="f1_pct",
        hue="domain_order",
        order=transitions,
        palette=["#6baed6", "#fd8d3c"],
        ax=ax,
    )

    ax.set_xlabel("Decoder MoE transition")
    ax.set_ylabel("Expert-set prediction F1 (%)")
    ax.set_title(f"Shifted mixed workload is asymmetric across mix order (cache ratio = {ratio:g})")
    ax.legend(title="Intra-request order", frameon=True, loc="best")
    fig.tight_layout()
    fig.savefig(figure_path, dpi=220)
    plt.close(fig)
    return figure_path


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    latency = load_latency(args.results_root)
    aggregated, layerwise = aggregate_confusion(args.results_root)

    aggregated = aggregated.sort_values(["condition", "cache_ratio", "cohort"]).reset_index(drop=True)
    layerwise = layerwise.sort_values(["condition", "cache_ratio", "cohort", "source_layer"]).reset_index(drop=True)

    save_table(latency, args.output_dir / "latency_compare_enriched.csv")
    save_table(aggregated, args.output_dir / "aggregated_confusion_boundary8.csv")
    save_table(layerwise, args.output_dir / "layerwise_confusion_boundary8.csv")

    figure_paths = [
        plot_latency_ratio(latency, args.output_dir),
        plot_f1_vs_cache(aggregated, args.output_dir),
        plot_layerwise_mixed(layerwise, args.layerwise_cache_ratios, args.output_dir),
        plot_shifted_mixed_domain_order(layerwise, args.domain_order_cache_ratio, args.output_dir),
    ]

    for figure_path in figure_paths:
        print(figure_path)


if __name__ == "__main__":
    main()
