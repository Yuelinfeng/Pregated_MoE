import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


EFFECT_ORDER = [
    "stable_order_bias",
    "ratio_shift_effect",
    "order_shift_effect",
    "combined_ratio_and_order_effect",
]

EFFECT_LABELS = {
    "stable_order_bias": "Stable BA vs stable AB",
    "ratio_shift_effect": "Ratio shift only vs stable AB",
    "order_shift_effect": "Order shift only vs stable balanced",
    "combined_ratio_and_order_effect": "Ratio + order shift vs stable balanced",
}

EFFECT_COLORS = {
    "stable_order_bias": "#8b8f97",
    "ratio_shift_effect": "#5c88c4",
    "order_shift_effect": "#d98b3a",
    "combined_ratio_and_order_effect": "#c84137",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--results_root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path)
    parser.add_argument("--cache_ratio", type=float, default=0.1)
    return parser.parse_args()


def load_effects(results_root: Path, cache_ratio: float) -> pd.DataFrame:
    path = results_root / "ablation_analysis" / "ratio_order_ablation_effects.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing ratio/order effect table: {path}")
    frame = pd.read_csv(path)
    frame["cache_ratio"] = frame["cache_ratio"].astype(float)
    subset = frame[frame["cache_ratio"] == cache_ratio].copy()
    subset = subset[subset["effect_name"].isin(EFFECT_ORDER)].copy()
    if subset.empty:
        raise RuntimeError(f"No ratio/order ablation rows found for cache_ratio={cache_ratio:g}")
    subset["effect_name"] = pd.Categorical(subset["effect_name"], categories=EFFECT_ORDER, ordered=True)
    subset = subset.sort_values("effect_name").reset_index(drop=True)
    subset["effect_label"] = subset["effect_name"].map(EFFECT_LABELS)
    subset["latency_delta_pp"] = subset["latency_ratio_delta"] * 100.0
    return subset


def load_combined_margin(results_root: Path, cache_ratio: float) -> pd.Series:
    path = results_root / "ablation_analysis" / "ratio_order_ablation_effects.csv"
    frame = pd.read_csv(path)
    frame["cache_ratio"] = frame["cache_ratio"].astype(float)
    subset = frame[
        (frame["cache_ratio"] == cache_ratio)
        & (frame["effect_name"] == "combined_minus_strongest_single")
    ]
    if subset.empty:
        return pd.Series(dtype=float)
    return subset.iloc[0]


def plot_summary(effects: pd.DataFrame, combined_margin: pd.Series, output_path: Path, cache_ratio: float) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=(12.8, 7.2))

    y_positions = list(range(len(effects)))
    colors = [EFFECT_COLORS[name] for name in effects["effect_name"]]
    bars = ax.barh(
        y_positions,
        effects["latency_delta_pp"],
        color=colors,
        edgecolor="white",
        linewidth=1.6,
        height=0.68,
    )

    ax.axvline(0.0, color="#30343b", linewidth=1.6, linestyle="--", alpha=0.9)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(effects["effect_label"])
    ax.invert_yaxis()
    ax.set_xlabel("Change in latency ratio vs matched stable control (percentage points)")
    ax.set_ylabel("")

    delta_min = min(effects["latency_delta_pp"].min(), -1.5)
    delta_max = max(effects["latency_delta_pp"].max(), 4.5)
    ax.set_xlim(delta_min - 0.5, delta_max + 0.85)

    for bar, (_, row) in zip(bars, effects.iterrows()):
        x = bar.get_width()
        y = bar.get_y() + bar.get_height() / 2
        sign = "+" if row["latency_delta_pp"] >= 0 else ""
        label = f"{sign}{row['latency_delta_pp']:.2f} pp\n{sign}{row['mean_delta_ms_gap']:.1f} ms"
        if x >= 0:
            ax.text(x + 0.12, y, label, va="center", ha="left", fontsize=12, color="#222222")
        else:
            ax.text(x - 0.12, y, label, va="center", ha="right", fontsize=12, color="#222222")

    fig.suptitle(
        "Order shift is the dominant single-axis stressor; ratio shift amplifies the combined case",
        y=0.965,
        fontsize=18,
    )
    fig.text(
        0.125,
        0.922,
        (
            f"Ratio/order ablation at cache ratio = {cache_ratio:g}. Positive values mean prefetch is slower "
            "than its matched stable control."
        ),
        fontsize=11.25,
        color="#444444",
    )

    ax.text(
        0.0,
        1.02,
        "0 = no extra harm over the matched stable control",
        transform=ax.transAxes,
        fontsize=10.5,
        color="#555555",
        ha="left",
        va="bottom",
    )

    if not combined_margin.empty:
        sign = "+" if combined_margin["latency_ratio_delta"] >= 0 else ""
        note = (
            "Combined stressor exceeds the stronger single-axis ablation by "
            f"{sign}{combined_margin['latency_ratio_delta'] * 100.0:.2f} pp "
            f"({sign}{combined_margin['mean_delta_ms_gap']:.1f} ms)"
        )
        fig.text(
            0.985,
            0.04,
            note,
            ha="right",
            va="bottom",
            fontsize=11.25,
            color="#1f1f1f",
        )

    fig.tight_layout(rect=[0, 0.08, 1, 0.9])
    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or (args.results_root / "figures")
    output_path = output_dir / "ratio_order_ablation_summary.png"

    effects = load_effects(args.results_root, args.cache_ratio)
    combined_margin = load_combined_margin(args.results_root, args.cache_ratio)
    plot_summary(effects, combined_margin, output_path, args.cache_ratio)
    print(output_path)


if __name__ == "__main__":
    main()
