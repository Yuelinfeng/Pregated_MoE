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


def parse_int_field(raw_value: Optional[str], default: int = -1) -> int:
    if raw_value is None or raw_value == "":
        return default
    return int(raw_value)


def parse_float_field(raw_value: Optional[str], default: float = math.nan) -> float:
    if raw_value is None or raw_value == "":
        return default
    return float(raw_value)


def parse_bool_field(raw_value: Optional[str]) -> bool:
    return parse_int_field(raw_value, 0) == 1


def iter_confusion_files(results_root: Path) -> Iterable[Tuple[str, float, Path]]:
    for csv_path in sorted((results_root / "prefetch").glob("*/cr*/confusion_boundary8.csv")):
        yield csv_path.parts[-3], parse_cache_ratio(csv_path.parts[-2]), csv_path


def iter_trace_files(results_root: Path) -> Iterable[Tuple[str, float, Path]]:
    for trace_path in sorted((results_root / "prefetch").glob("*/cr*/prefetch_trace.tsv")):
        yield trace_path.parts[-3], parse_cache_ratio(trace_path.parts[-2]), trace_path


def iter_trace_rows(trace_path: Path) -> Iterable[Dict[str, str]]:
    with trace_path.open("r", encoding="utf-8-sig") as trace_file:
        reader = csv.DictReader(trace_file, delimiter="\t")
        for row in reader:
            yield row


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
    for row in iter_trace_rows(trace_path):
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


def extract_prefetch_utility_events(condition: str, cache_ratio: float, trace_path: Path) -> List[Dict[str, object]]:
    actual_by_issue: Dict[int, List[int]] = {}
    actual_count_by_issue: Dict[int, int] = {}
    layer_by_issue: Dict[int, Tuple[str, str]] = {}

    trace_rows = list(iter_trace_rows(trace_path))
    for row in trace_rows:
        if row["event_type"] != "CONFUSION":
            continue
        prefetch_issue_id = parse_int_field(row.get("prefetch_issue_id"), -1)
        if prefetch_issue_id < 0:
            continue
        actual_experts = parse_expert_list(row["actual_experts"])
        actual_by_issue[prefetch_issue_id] = actual_experts
        actual_count_by_issue[prefetch_issue_id] = len(actual_experts)
        layer_by_issue[prefetch_issue_id] = (row["source_layer"], row["target_layer"])

    utility_events: List[Dict[str, object]] = []
    for row in trace_rows:
        if row["event_type"] != "PREFETCH_EXPERT":
            continue
        prefetch_issue_id = parse_int_field(row.get("prefetch_issue_id"), -1)
        expert_id = parse_int_field(row.get("expert_id"), -1)
        if prefetch_issue_id < 0 or expert_id < 0 or prefetch_issue_id not in actual_by_issue:
            continue

        source_layer, target_layer = layer_by_issue[prefetch_issue_id]
        actual_experts = set(actual_by_issue[prefetch_issue_id])
        useful = expert_id in actual_experts
        timely = parse_bool_field(row.get("ready_before_consume"))
        cache_hit = parse_bool_field(row.get("cache_hit"))

        utility_events.append(
            {
                "condition": condition,
                "cache_ratio": cache_ratio,
                "source_layer": source_layer,
                "target_layer": target_layer,
                "prefetch_issue_id": prefetch_issue_id,
                "expert_id": expert_id,
                "cache_hit": int(cache_hit),
                "timely": int(timely),
                "late": int(not timely),
                "useful": int(useful),
                "useless": int(not useful),
                "timely_useful": int(timely and useful),
                "late_useful": int((not timely) and useful),
                "timely_useless": int(timely and (not useful)),
                "late_useless": int((not timely) and (not useful)),
                "stall_time_ms": parse_float_field(row.get("stall_time_ms"), 0.0),
                "actual_required_experts": actual_count_by_issue[prefetch_issue_id],
            }
        )

    return utility_events


def summarize_prefetch_utility(frame: pd.DataFrame, group_columns: List[str]) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()

    rows: List[Dict[str, object]] = []
    for group_key, group in frame.groupby(group_columns):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        row = dict(zip(group_columns, group_key))

        total_prefetch_experts = int(len(group))
        issue_stalls = group.groupby("prefetch_issue_id")["stall_time_ms"].max()
        issue_late_flags = group.groupby("prefetch_issue_id")["late"].max()
        issue_actual_counts = group.groupby("prefetch_issue_id")["actual_required_experts"].max()

        row.update(
            {
                "prefetch_experts": total_prefetch_experts,
                "prefetch_issues": int(issue_stalls.shape[0]),
                "cache_hit_experts": int(group["cache_hit"].sum()),
                "cache_miss_experts": int(total_prefetch_experts - group["cache_hit"].sum()),
                "timely_prefetch_experts": int(group["timely"].sum()),
                "late_prefetch_experts": int(group["late"].sum()),
                "useful_prefetch_experts": int(group["useful"].sum()),
                "useless_prefetch_experts": int(group["useless"].sum()),
                "timely_useful_prefetch_experts": int(group["timely_useful"].sum()),
                "late_useful_prefetch_experts": int(group["late_useful"].sum()),
                "timely_useless_prefetch_experts": int(group["timely_useless"].sum()),
                "late_useless_prefetch_experts": int(group["late_useless"].sum()),
                "timely_useful_prefetch_ratio": safe_divide(group["timely_useful"].sum(), total_prefetch_experts),
                "late_useful_prefetch_ratio": safe_divide(group["late_useful"].sum(), total_prefetch_experts),
                "useful_prefetch_ratio": safe_divide(group["useful"].sum(), total_prefetch_experts),
                "late_prefetch_ratio": safe_divide(group["late"].sum(), total_prefetch_experts),
                "late_prefetch_issue_ratio": safe_divide(issue_late_flags.sum(), len(issue_late_flags)),
                "avg_prefetch_stall_ms": float(issue_stalls.mean()) if not issue_stalls.empty else math.nan,
                "p95_prefetch_stall_ms": float(issue_stalls.quantile(0.95)) if not issue_stalls.empty else math.nan,
                "avg_actual_required_experts": float(issue_actual_counts.mean()) if not issue_actual_counts.empty else math.nan,
            }
        )
        rows.append(row)

    return pd.DataFrame(rows)


def aggregate_prefetch_utility(results_root: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    utility_rows: List[Dict[str, object]] = []
    for condition, cache_ratio, trace_path in iter_trace_files(results_root):
        utility_rows.extend(extract_prefetch_utility_events(condition, cache_ratio, trace_path))

    utility_frame = pd.DataFrame(utility_rows)
    if utility_frame.empty:
        return pd.DataFrame(), pd.DataFrame()

    summary = summarize_prefetch_utility(utility_frame, ["condition", "cache_ratio"])
    by_layer = summarize_prefetch_utility(utility_frame, ["condition", "cache_ratio", "source_layer", "target_layer"])
    return summary, by_layer


def factorial_interaction(latency: pd.DataFrame) -> pd.DataFrame:
    if latency.empty:
        return pd.DataFrame()

    available_conditions = set(latency["condition"].unique())
    shifted_mixed_condition = normalize_condition("shifted_mixed", available_conditions)

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
    utility_summary, utility_by_layer = aggregate_prefetch_utility(args.results_root)
    if not utility_summary.empty:
        mechanisms = mechanisms.merge(utility_summary, on=["condition", "cache_ratio"], how="left")
    mechanisms = add_latency_context(mechanisms, latency, args.high_hit_threshold)
    interaction = factorial_interaction(latency)
    transition_entropy, transition_divergence = analyze_transition_kernels(args.results_root, args.reference_conditions)

    write_frame(mechanisms, args.output_dir / "prefetch_mechanism_metrics.csv")
    write_frame(utility_by_layer, args.output_dir / "prefetch_utility_by_layer.csv")
    write_frame(interaction, args.output_dir / "factorial_interaction_effects.csv")
    write_frame(transition_entropy, args.output_dir / "transition_entropy.csv")
    write_frame(transition_divergence, args.output_dir / "transition_kernel_divergence.csv")


if __name__ == "__main__":
    main()
