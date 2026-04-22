import argparse
from pathlib import Path

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
    "stable_homogeneous": "Stable homogeneous",
    "shifted_homogeneous": "Shifted homogeneous",
    "stable_mixed": "Stable mixed",
    "shifted_mixed": "Shifted mixed",
}

DOMAIN_FAMILY = {
    "stable_homogeneous": "Homogeneous",
    "shifted_homogeneous": "Homogeneous",
    "stable_mixed": "Mixed",
    "shifted_mixed": "Mixed",
}

STREAM_STATE = {
    "stable_homogeneous": "Stable",
    "stable_mixed": "Stable",
    "shifted_homogeneous": "Shifted",
    "shifted_mixed": "Shifted",
}

FAMILY_COLORS = {
    "Homogeneous": "#2f5d8a",
    "Mixed": "#d17a22",
}

STATE_MARKERS = {
    "Stable": "o",
    "Shifted": "D",
}

CACHE_SIZE_MAP = {
    0.03: 120,
    0.10: 210,
    0.40: 340,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--results_root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    return parser.parse_args()


def load_summary(results_root: Path) -> pd.DataFrame:
    latency = pd.read_csv(results_root / "latency_compare.csv")
    mechanism = pd.read_csv(results_root / "formal_analysis" / "prefetch_mechanism_metrics.csv")
    summary = latency.merge(
        mechanism[
            [
                "condition",
                "cache_ratio",
                "avg_cache_hit_rate",
                "useful_prefetch_precision",
                "expert_set_f1",
                "high_hit_low_utility_failure",
            ]
        ],
        on=["condition", "cache_ratio"],
        how="inner",
    )
    summary["condition_label"] = summary["condition"].map(CONDITION_LABELS)
    summary["domain_family"] = summary["condition"].map(DOMAIN_FAMILY)
    summary["stream_state"] = summary["condition"].map(STREAM_STATE)
    summary["cache_ratio"] = summary["cache_ratio"].astype(float)
    summary["cache_ratio_label"] = summary["cache_ratio"].map(
        {
            0.03: "0.03",
            0.1: "0.10",
            0.4: "0.40",
        }
    )
    summary["cache_hit_pct"] = summary["avg_cache_hit_rate"].astype(float) * 100.0
    summary["latency_ratio"] = summary["prefetch_over_on_demand"].astype(float)
    summary["delta_ms"] = summary["prefetch_mean_ms"].astype(float) - summary["on_demand_mean_ms"].astype(float)
    return summary.sort_values(["condition", "cache_ratio"]).reset_index(drop=True)


def build_condition_legend_handles():
    handles = []
    labels = []
    for condition in CONDITION_ORDER:
        handle = plt.Line2D(
            [],
            [],
            color=FAMILY_COLORS[DOMAIN_FAMILY[condition]],
            marker=STATE_MARKERS[STREAM_STATE[condition]],
            linestyle="None",
            markersize=9,
            markeredgecolor="white",
            markeredgewidth=1.1,
        )
        handles.append(handle)
        labels.append(CONDITION_LABELS[condition])
    return handles, labels


def build_cache_legend_handles():
    handles = []
    labels = []
    for cache_ratio in [0.03, 0.10, 0.40]:
        ms = (CACHE_SIZE_MAP[cache_ratio] ** 0.5) * 0.75
        handle = plt.Line2D(
            [],
            [],
            color="#737373",
            marker="o",
            linestyle="None",
            markersize=ms,
            markerfacecolor="#d9d9d9",
            markeredgecolor="#737373",
            markeredgewidth=1.0,
            alpha=0.95,
        )
        handles.append(handle)
        labels.append(f"cache_ratio={cache_ratio:.2f}")
    return handles, labels


def plot_summary(summary: pd.DataFrame, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "prefetch_shift_main_summary.png"

    sns.set_theme(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=(11.0, 7.4))

    # Highlight the "high-hit, low-utility" region without overclaiming a threshold.
    ax.axhspan(1.0, 1.115, color="#f3d9cf", alpha=0.55, zorder=0)
    ax.axvspan(90.0, 98.0, color="#f6f6f6", alpha=0.55, zorder=0)
    ax.axhline(1.0, color="#444444", linestyle="--", linewidth=1.3, zorder=1)
    ax.text(84.2, 1.0015, "break-even", fontsize=10.5, color="#444444", ha="left", va="bottom")
    ax.text(
        96.9,
        1.108,
        "High-hit, low-utility region",
        fontsize=10.5,
        color="#7a3e2b",
        ha="right",
        va="top",
    )

    for condition in CONDITION_ORDER:
        subset = summary[summary["condition"] == condition].sort_values("cache_ratio")
        color = FAMILY_COLORS[DOMAIN_FAMILY[condition]]
        marker = STATE_MARKERS[STREAM_STATE[condition]]

        ax.plot(
            subset["cache_hit_pct"],
            subset["latency_ratio"],
            color=color,
            linewidth=1.7,
            alpha=0.75,
            zorder=2,
        )

        ax.scatter(
            subset["cache_hit_pct"],
            subset["latency_ratio"],
            s=[CACHE_SIZE_MAP[value] for value in subset["cache_ratio"]],
            color=color,
            marker=marker,
            alpha=0.95,
            edgecolor="white",
            linewidth=1.4,
            zorder=3,
        )

        for _, row in subset.iterrows():
            dx = 0.18 if row["stream_state"] == "Stable" else 0.22
            dy = 0.0015 if row["domain_family"] == "Homogeneous" else -0.002
            ax.text(
                row["cache_hit_pct"] + dx,
                row["latency_ratio"] + dy,
                row["cache_ratio_label"],
                fontsize=9.5,
                color=color,
                ha="left",
                va="center",
                zorder=4,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 0.2},
            )

    # Short condition labels near the highest-cache points to reduce legend chasing.
    for condition in CONDITION_ORDER:
        subset = summary[summary["condition"] == condition].sort_values("cache_ratio")
        row = subset.iloc[-1]
        label_x = row["cache_hit_pct"] + 0.28
        label_y = row["latency_ratio"]
        if condition == "shifted_mixed":
            label_y += 0.0048
        elif condition == "stable_mixed":
            label_y -= 0.001
        elif condition == "shifted_homogeneous":
            label_y -= 0.0042
        else:
            label_y += 0.0008
        ax.text(
            label_x,
            label_y,
            CONDITION_LABELS[condition],
            fontsize=10.5,
            color=FAMILY_COLORS[DOMAIN_FAMILY[condition]],
            ha="left",
            va="center",
            weight="medium",
        )

    ax.set_xlim(84.0, 98.0)
    ax.set_ylim(0.975, 1.115)
    ax.set_xlabel("Average cache hit rate (%)")
    ax.set_ylabel("Latency ratio (prefetch / on-demand)")
    ax.set_title(
        "Full main experiment\nHigh cache hit rate does not rescue decoder prefetch under shifting mixed workloads",
        pad=14,
    )

    condition_handles, condition_labels = build_condition_legend_handles()
    cache_handles, cache_labels = build_cache_legend_handles()

    legend1 = ax.legend(
        condition_handles,
        condition_labels,
        title="Workload condition",
        loc="upper left",
        frameon=True,
        fontsize=10.5,
        title_fontsize=10.5,
    )
    ax.add_artist(legend1)
    ax.legend(
        cache_handles,
        cache_labels,
        title="Marker size",
        loc="lower right",
        frameon=True,
        fontsize=10.0,
        title_fontsize=10.0,
    )

    fig.text(
        0.5,
        0.02,
        "Color encodes homogeneous vs mixed requests; marker shape encodes stable vs shifted streams; point size encodes cache ratio.",
        ha="center",
        va="center",
        fontsize=10.5,
    )
    fig.tight_layout(rect=[0, 0.05, 1, 0.98])
    fig.savefig(output_path, dpi=260)
    plt.close(fig)
    return output_path


def main() -> None:
    args = parse_args()
    summary = load_summary(args.results_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_dir / "prefetch_shift_main_summary_data.csv", index=False)
    output_path = plot_summary(summary, args.output_dir)
    print(output_path)


if __name__ == "__main__":
    main()
