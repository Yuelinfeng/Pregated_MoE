import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd


ABLATION_CONDITIONS = [
    "stable_mixed_ab",
    "stable_mixed_ba",
    "stable_mixed_balanced_order",
    "shifted_mixed_ratio_only",
    "shifted_mixed_order_only",
    "shifted_mixed_ratio_and_order",
]


def load_latency(results_root: Path) -> pd.DataFrame:
    path = results_root / "latency_compare.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing latency table: {path}")
    frame = pd.read_csv(path)
    frame["cache_ratio"] = frame["cache_ratio"].astype(float)
    frame["prefetch_over_on_demand"] = frame["prefetch_over_on_demand"].astype(float)
    frame["mean_delta_ms"] = frame["prefetch_mean_ms"].astype(float) - frame["on_demand_mean_ms"].astype(float)
    frame["p95_delta_ms"] = frame["prefetch_p95_ms"].astype(float) - frame["on_demand_p95_ms"].astype(float)
    return frame


def load_mechanisms(results_root: Path) -> pd.DataFrame:
    path = results_root / "formal_analysis" / "prefetch_mechanism_metrics.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing mechanism table: {path}")
    frame = pd.read_csv(path)
    frame["cache_ratio"] = frame["cache_ratio"].astype(float)
    return frame


def build_summary(latency: pd.DataFrame, mechanisms: pd.DataFrame) -> pd.DataFrame:
    keep = [
        "condition",
        "cache_ratio",
        "avg_cache_hit_rate",
        "useful_prefetch_precision",
        "expert_set_f1",
        "wasted_prefetch_experts",
        "wasted_prefetch_bytes",
        "high_hit_low_utility_failure",
    ]
    merged = latency.merge(mechanisms[keep], on=["condition", "cache_ratio"], how="left")
    merged = merged[merged["condition"].isin(ABLATION_CONDITIONS)].copy()
    merged = merged.sort_values(["cache_ratio", "condition"]).reset_index(drop=True)
    return merged


def compute_effect_rows(summary: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for cache_ratio, group in summary.groupby("cache_ratio"):
        values = {
            row["condition"]: row
            for _, row in group.iterrows()
        }

        required = [
            "stable_mixed_ab",
            "stable_mixed_ba",
            "stable_mixed_balanced_order",
            "shifted_mixed_ratio_only",
            "shifted_mixed_order_only",
            "shifted_mixed_ratio_and_order",
        ]
        if any(condition not in values for condition in required):
            continue

        def effect_row(
            effect_name: str,
            shifted_condition: str,
            baseline_condition: str,
            rationale: str,
        ) -> Dict[str, object]:
            shifted = values[shifted_condition]
            baseline = values[baseline_condition]
            return {
                "cache_ratio": cache_ratio,
                "effect_name": effect_name,
                "shifted_condition": shifted_condition,
                "baseline_condition": baseline_condition,
                "rationale": rationale,
                "shifted_latency_ratio": shifted["prefetch_over_on_demand"],
                "baseline_latency_ratio": baseline["prefetch_over_on_demand"],
                "latency_ratio_delta": shifted["prefetch_over_on_demand"] - baseline["prefetch_over_on_demand"],
                "shifted_mean_delta_ms": shifted["mean_delta_ms"],
                "baseline_mean_delta_ms": baseline["mean_delta_ms"],
                "mean_delta_ms_gap": shifted["mean_delta_ms"] - baseline["mean_delta_ms"],
                "shifted_f1": shifted["expert_set_f1"],
                "baseline_f1": baseline["expert_set_f1"],
                "f1_delta": shifted["expert_set_f1"] - baseline["expert_set_f1"],
                "shifted_hit_rate": shifted["avg_cache_hit_rate"],
                "baseline_hit_rate": baseline["avg_cache_hit_rate"],
                "hit_rate_delta": shifted["avg_cache_hit_rate"] - baseline["avg_cache_hit_rate"],
            }

        rows.append(
            effect_row(
                "stable_order_bias",
                "stable_mixed_ba",
                "stable_mixed_ab",
                "Stable-order sanity check: does merely reversing AB to BA change behavior when nothing shifts over time?",
            )
        )
        rows.append(
            effect_row(
                "ratio_shift_effect",
                "shifted_mixed_ratio_only",
                "stable_mixed_ab",
                "Keep request order fixed as AB and isolate only the stream-level ratio shift.",
            )
        )
        rows.append(
            effect_row(
                "order_shift_effect",
                "shifted_mixed_order_only",
                "stable_mixed_balanced_order",
                "Keep the 50/50 mix ratio fixed and isolate only the AB/BA order alternation across phases.",
            )
        )
        rows.append(
            effect_row(
                "combined_ratio_and_order_effect",
                "shifted_mixed_ratio_and_order",
                "stable_mixed_balanced_order",
                "Combined stressor: both mix ratio and order shift across phases.",
            )
        )

        combined = values["shifted_mixed_ratio_and_order"]["prefetch_over_on_demand"]
        strongest_single = max(
            values["shifted_mixed_ratio_only"]["prefetch_over_on_demand"],
            values["shifted_mixed_order_only"]["prefetch_over_on_demand"],
        )
        rows.append(
            {
                "cache_ratio": cache_ratio,
                "effect_name": "combined_minus_strongest_single",
                "shifted_condition": "shifted_mixed_ratio_and_order",
                "baseline_condition": "max(shifted_mixed_ratio_only, shifted_mixed_order_only)",
                "rationale": "Does combining ratio and order shift exceed the stronger single-axis ablation?",
                "shifted_latency_ratio": combined,
                "baseline_latency_ratio": strongest_single,
                "latency_ratio_delta": combined - strongest_single,
                "shifted_mean_delta_ms": values["shifted_mixed_ratio_and_order"]["mean_delta_ms"],
                "baseline_mean_delta_ms": max(
                    values["shifted_mixed_ratio_only"]["mean_delta_ms"],
                    values["shifted_mixed_order_only"]["mean_delta_ms"],
                ),
                "mean_delta_ms_gap": values["shifted_mixed_ratio_and_order"]["mean_delta_ms"]
                - max(
                    values["shifted_mixed_ratio_only"]["mean_delta_ms"],
                    values["shifted_mixed_order_only"]["mean_delta_ms"],
                ),
                "shifted_f1": values["shifted_mixed_ratio_and_order"]["expert_set_f1"],
                "baseline_f1": min(
                    values["shifted_mixed_ratio_only"]["expert_set_f1"],
                    values["shifted_mixed_order_only"]["expert_set_f1"],
                ),
                "f1_delta": values["shifted_mixed_ratio_and_order"]["expert_set_f1"]
                - min(
                    values["shifted_mixed_ratio_only"]["expert_set_f1"],
                    values["shifted_mixed_order_only"]["expert_set_f1"],
                ),
                "shifted_hit_rate": values["shifted_mixed_ratio_and_order"]["avg_cache_hit_rate"],
                "baseline_hit_rate": min(
                    values["shifted_mixed_ratio_only"]["avg_cache_hit_rate"],
                    values["shifted_mixed_order_only"]["avg_cache_hit_rate"],
                ),
                "hit_rate_delta": values["shifted_mixed_ratio_and_order"]["avg_cache_hit_rate"]
                - min(
                    values["shifted_mixed_ratio_only"]["avg_cache_hit_rate"],
                    values["shifted_mixed_order_only"]["avg_cache_hit_rate"],
                ),
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--results_root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path)
    args = parser.parse_args()

    output_dir = args.output_dir or (args.results_root / "ablation_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    latency = load_latency(args.results_root)
    mechanisms = load_mechanisms(args.results_root)
    summary = build_summary(latency, mechanisms)
    effects = compute_effect_rows(summary)

    summary_path = output_dir / "ratio_order_ablation_summary.csv"
    effects_path = output_dir / "ratio_order_ablation_effects.csv"
    summary.to_csv(summary_path, index=False)
    effects.to_csv(effects_path, index=False)

    print(summary_path)
    print(summary.to_string(index=False))
    print()
    print(effects_path)
    if effects.empty:
        print("No complete ratio/order ablation grid found.")
    else:
        print(effects.to_string(index=False))


if __name__ == "__main__":
    main()
