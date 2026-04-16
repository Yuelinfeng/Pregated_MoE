import argparse
import csv
import math
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd


MAIN_CONDITIONS = [
    "stable_homogeneous",
    "shifted_homogeneous",
    "stable_mixed",
    "shifted_mixed",
]

SHIFTED_MIXED_FALLBACKS = [
    "shifted_mixed",
    "shifted_mixed_ratio_and_order",
    "shifted_mixed_ratio_only",
    "shifted_mixed_order_only",
]


def safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def parse_expert_list(raw_value: str) -> List[int]:
    if not raw_value:
        return []
    return [int(token) for token in raw_value.split(",") if token]


def parse_cache_ratio(raw_value: str) -> float:
    return float(raw_value.replace("cr", "", 1))


def iter_confusion_files(results_root: Path) -> Iterable[Tuple[str, float, Path]]:
    for csv_path in sorted((results_root / "prefetch").glob("*/cr*/confusion_boundary8.csv")):
        yield csv_path.parts[-3], parse_cache_ratio(csv_path.parts[-2]), csv_path


def iter_trace_files(results_root: Path) -> Iterable[Tuple[str, float, Path]]:
    for trace_path in sorted((results_root / "prefetch").glob("*/cr*/prefetch_trace.tsv")):
        yield trace_path.parts[-3], parse_cache_ratio(trace_path.parts[-2]), trace_path


def normalize_condition(condition: str, available_conditions: Iterable[str]) -> str:
    available = set(available_conditions)
    if condition != "shifted_mixed":
        return condition
    for candidate in SHIFTED_MIXED_FALLBACKS:
        if candidate in available:
            return candidate
    return condition


def aggregate_confusion(results_root: Path, expert_bytes: int) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for condition, cache_ratio, csv_path in iter_confusion_files(results_root):
        frame = pd.read_csv(csv_path)
        subset = frame[frame["cohort"] == "all"]
        if subset.empty:
            continue

        tp = int(subset["tp"].sum())
        fp = int(subset["fp"].sum())
        fn = int(subset["fn"].sum())
        tn = int(subset["tn"].sum())
        precision = safe_divide(tp, tp + fp)
        recall = safe_divide(tp, tp + fn)
        f1 = safe_divide(2.0 * precision * recall, precision + recall)
        rows.append(
            {
                "condition": condition,
                "cache_ratio": cache_ratio,
                "events": int(subset["events"].sum()),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "useful_prefetch_precision": precision,
                "expert_set_recall": recall,
                "expert_set_f1": f1,
                "wasted_prefetch_experts": fp,
                "missed_actual_experts": fn,
                "wasted_prefetch_bytes": fp * expert_bytes,
                "avg_cache_hit_rate": float(subset["avg_cache_hit_rate"].mean()),
                "avg_active_experts": float(subset["avg_active_experts"].mean()),
                "avg_max_active_experts": float(subset["avg_max_active_experts"].mean()),
            }
        )
    return pd.DataFrame(rows)


def load_latency(latency_csv: Optional[Path], results_root: Path) -> pd.DataFrame:
    path = latency_csv or (results_root / "latency_compare.csv")
    if not path.is_file():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    frame["cache_ratio"] = frame["cache_ratio"].astype(float)
    frame["latency_ratio"] = frame["prefetch_over_on_demand"].astype(float)
    frame["delta_ms"] = frame["prefetch_mean_ms"].astype(float) - frame["on_demand_mean_ms"].astype(float)
    return frame


def add_latency_context(mechanisms: pd.DataFrame, latency: pd.DataFrame, high_hit_threshold: float) -> pd.DataFrame:
    if mechanisms.empty or latency.empty:
        return mechanisms
    merged = mechanisms.merge(
        latency[["condition", "cache_ratio", "latency_ratio", "delta_ms"]],
        on=["condition", "cache_ratio"],
        how="left",
    )
    merged["high_hit_low_utility_failure"] = (
        (merged["avg_cache_hit_rate"] >= high_hit_threshold)
        & (merged["latency_ratio"] > 1.0)
    ).astype(int)
    return merged


def transition_distribution_from_trace(trace_path: Path) -> Dict[Tuple[str, str], Counter]:
    kernels: Dict[Tuple[str, str], Counter] = {}
    with trace_path.open("r", encoding="utf-8-sig") as trace_file:
        reader = csv.DictReader(trace_file, delimiter="\t")
        for row in reader:
            if row["event_type"] != "CONFUSION":
                continue
            key = (row["source_layer"], row["target_layer"])
            predicted = parse_expert_list(row["predicted_experts"])
            actual = parse_expert_list(row["actual_experts"])
            kernel = kernels.setdefault(key, Counter())
            for source_expert in predicted:
                for target_expert in actual:
                    kernel[(source_expert, target_expert)] += 1
    return kernels


def entropy_bits(counter: Counter) -> float:
    total = sum(counter.values())
    if total == 0:
        return 0.0
    entropy = 0.0
    for count in counter.values():
        probability = count / total
        entropy -= probability * math.log2(probability)
    return entropy


def js_divergence_bits(left: Counter, right: Counter) -> float:
    left_total = sum(left.values())
    right_total = sum(right.values())
    if left_total == 0 or right_total == 0:
        return math.nan

    keys = set(left) | set(right)

    def probability(counter: Counter, total: int, key: Tuple[int, int]) -> float:
        return counter.get(key, 0) / total

    divergence = 0.0
    for key in keys:
        p = probability(left, left_total, key)
        q = probability(right, right_total, key)
        m = 0.5 * (p + q)
        if p > 0:
            divergence += 0.5 * p * math.log2(p / m)
        if q > 0:
            divergence += 0.5 * q * math.log2(q / m)
    return divergence


def analyze_transition_kernels(results_root: Path, reference_conditions: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    kernels: Dict[Tuple[str, float, str, str], Counter] = {}
    entropy_rows: List[Dict[str, object]] = []

    for condition, cache_ratio, trace_path in iter_trace_files(results_root):
        for (source_layer, target_layer), counter in transition_distribution_from_trace(trace_path).items():
            key = (condition, cache_ratio, source_layer, target_layer)
            kernels[key] = counter
            entropy_rows.append(
                {
                    "condition": condition,
                    "cache_ratio": cache_ratio,
                    "source_layer": source_layer,
                    "target_layer": target_layer,
                    "transition_events": sum(counter.values()),
                    "transition_entropy_bits": entropy_bits(counter),
                }
            )

    available_conditions = {key[0] for key in kernels}
    divergence_rows: List[Dict[str, object]] = []
    for key, target_counter in sorted(kernels.items()):
        condition, cache_ratio, source_layer, target_layer = key
        for reference_condition in reference_conditions:
            normalized_reference = normalize_condition(reference_condition, available_conditions)
            reference_key = (normalized_reference, cache_ratio, source_layer, target_layer)
            if reference_key not in kernels or reference_key == key:
                continue
            divergence_rows.append(
                {
                    "condition": condition,
                    "cache_ratio": cache_ratio,
                    "source_layer": source_layer,
                    "target_layer": target_layer,
                    "reference_condition": normalized_reference,
                    "transition_js_divergence_bits": js_divergence_bits(kernels[reference_key], target_counter),
                }
            )

    return pd.DataFrame(entropy_rows), pd.DataFrame(divergence_rows)


def factorial_interaction(latency: pd.DataFrame) -> pd.DataFrame:
    if latency.empty:
        return pd.DataFrame()

    available_conditions = set(latency["condition"].unique())
    shifted_mixed_condition = normalize_condition("shifted_mixed", available_conditions)
    condition_map = {
        "stable_homogeneous": (0, 0),
        "shifted_homogeneous": (1, 0),
        "stable_mixed": (0, 1),
        shifted_mixed_condition: (1, 1),
    }

    rows: List[Dict[str, object]] = []
    group_columns = ["cache_ratio"]
    if "seed" in latency.columns:
        group_columns.append("seed")

    for group_key, group in latency.groupby(group_columns):
        values = {row["condition"]: float(row["latency_ratio"]) for _, row in group.iterrows()}
        required = ["stable_homogeneous", "shifted_homogeneous", "stable_mixed", shifted_mixed_condition]
        if any(condition not in values for condition in required):
            continue

        interaction = (
            values[shifted_mixed_condition]
            - values["shifted_homogeneous"]
            - values["stable_mixed"]
            + values["stable_homogeneous"]
        )
        cache_ratio = group_key[0] if isinstance(group_key, tuple) else group_key
        seed = group_key[1] if isinstance(group_key, tuple) and len(group_key) > 1 else None
        rows.append(
            {
                "cache_ratio": cache_ratio,
                "seed": seed,
                "stable_homogeneous": values["stable_homogeneous"],
                "shifted_homogeneous": values["shifted_homogeneous"],
                "stable_mixed": values["stable_mixed"],
                "shifted_mixed": values[shifted_mixed_condition],
                "shifted_mixed_source_condition": shifted_mixed_condition,
                "interaction_beta3": interaction,
            }
        )

    return pd.DataFrame(rows)


def write_frame(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--results_root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--latency_csv", type=Path)
    parser.add_argument("--expert_bytes", type=int, default=18874368)
    parser.add_argument("--high_hit_threshold", type=float, default=0.85)
    parser.add_argument("--reference_conditions", nargs="*", default=["stable_homogeneous", "stable_mixed"])
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    latency = load_latency(args.latency_csv, args.results_root)
    mechanisms = aggregate_confusion(args.results_root, args.expert_bytes)
    mechanisms = add_latency_context(mechanisms, latency, args.high_hit_threshold)
    interaction = factorial_interaction(latency)
    transition_entropy, transition_divergence = analyze_transition_kernels(args.results_root, args.reference_conditions)

    write_frame(mechanisms, args.output_dir / "prefetch_mechanism_metrics.csv")
    write_frame(interaction, args.output_dir / "factorial_interaction_effects.csv")
    write_frame(transition_entropy, args.output_dir / "transition_entropy.csv")
    write_frame(transition_divergence, args.output_dir / "transition_kernel_divergence.csv")


if __name__ == "__main__":
    main()
