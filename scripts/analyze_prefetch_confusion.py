import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


def safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def parse_expert_list(raw_value: str) -> List[int]:
    if not raw_value:
        return []
    return [int(token) for token in raw_value.split(",") if token]


def parse_expert_counts(raw_value: str) -> Dict[int, int]:
    counts: Dict[int, int] = {}
    if not raw_value:
        return counts
    for token in raw_value.split(","):
        if not token:
            continue
        expert, count = token.split(":")
        counts[int(expert)] = int(count)
    return counts


def iter_trace_rows(trace_path: Path) -> Iterable[Dict[str, str]]:
    with trace_path.open("r", encoding="utf-8-sig") as trace_file:
        reader = csv.DictReader(trace_file, delimiter="\t")
        for row in reader:
            yield row


def load_request_metadata(request_metrics_path: Optional[Path]) -> Dict[str, Dict[str, object]]:
    if request_metrics_path is None:
        return {}

    request_metadata: Dict[str, Dict[str, object]] = {}
    with request_metrics_path.open("r", encoding="utf-8-sig") as metrics_file:
        for line in metrics_file:
            if not line.strip():
                continue
            payload = json.loads(line)
            request_metadata[payload["request_id"]] = payload
    return request_metadata


def cohorts_for_request(request_info: Optional[Dict[str, object]], boundary_window: int) -> List[str]:
    cohorts = ["all"]
    if not request_info:
        return cohorts

    domain_label = request_info.get("domain_label")
    if domain_label:
        cohorts.append(f"domain:{domain_label}")

    metadata = request_info.get("metadata", {})
    if metadata.get("is_shift_boundary"):
        cohorts.append("shift_boundary")

    if boundary_window > 0 and metadata.get("distance_to_nearest_boundary", boundary_window + 1) <= boundary_window:
        cohorts.append(f"boundary_window_{boundary_window}")

    return cohorts


def aggregate_confusion(
    trace_path: Path,
    request_metadata: Dict[str, Dict[str, object]],
    boundary_window: int,
) -> List[Dict[str, object]]:
    aggregates: Dict[Tuple[str, str, str, str], Dict[str, object]] = defaultdict(
        lambda: {
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "tn": 0,
            "weighted_tp": 0,
            "weighted_fn": 0,
            "weighted_total_actual": 0,
            "events": 0,
        }
    )

    request_summaries: Dict[Tuple[str, str], Dict[str, float]] = defaultdict(
        lambda: {"count": 0, "cache_hit_rate_sum": 0.0, "max_active_experts_sum": 0.0, "avg_active_experts_sum": 0.0}
    )

    for row in iter_trace_rows(trace_path):
        event_type = row["event_type"]
        condition = row["condition"]
        source_layer = row["source_layer"]
        target_layer = row["target_layer"]
        request_id = row["request_id"]
        cohorts = cohorts_for_request(request_metadata.get(request_id), boundary_window)

        if event_type == "CONFUSION":
            predicted = set(parse_expert_list(row["predicted_experts"]))
            actual_counts = parse_expert_counts(row["actual_counts"])
            weighted_tp = 0
            weighted_fn = 0
            weighted_total_actual = 0
            for expert, count in actual_counts.items():
                weighted_total_actual += count
                if expert in predicted:
                    weighted_tp += count
                else:
                    weighted_fn += count

            for cohort in cohorts:
                key = (condition, cohort, source_layer, target_layer)
                aggregate = aggregates[key]
                aggregate["tp"] += int(row["tp"])
                aggregate["fp"] += int(row["fp"])
                aggregate["fn"] += int(row["fn"])
                aggregate["tn"] += int(row["tn"])
                aggregate["events"] += 1
                aggregate["weighted_tp"] += weighted_tp
                aggregate["weighted_fn"] += weighted_fn
                aggregate["weighted_total_actual"] += weighted_total_actual

        elif event_type == "SUMMARY":
            for cohort in cohorts:
                summary = request_summaries[(condition, cohort)]
                summary["count"] += 1
                summary["cache_hit_rate_sum"] += float(row["source_layer"])
                summary["max_active_experts_sum"] += float(row["target_layer"])
                summary["avg_active_experts_sum"] += float(row["num_experts"])

    results: List[Dict[str, object]] = []
    for (condition, cohort, source_layer, target_layer), aggregate in sorted(aggregates.items()):
        tp = aggregate["tp"]
        fp = aggregate["fp"]
        fn = aggregate["fn"]
        tn = aggregate["tn"]
        precision = safe_divide(tp, tp + fp)
        recall = safe_divide(tp, tp + fn)
        f1 = safe_divide(2 * precision * recall, precision + recall)
        specificity = safe_divide(tn, tn + fp)
        balanced_accuracy = 0.5 * (recall + specificity)
        weighted_recall = safe_divide(aggregate["weighted_tp"], aggregate["weighted_total_actual"])

        summary = request_summaries[(condition, cohort)]
        summary_count = summary["count"] if summary["count"] else 1

        results.append(
            {
                "condition": condition,
                "cohort": cohort,
                "source_layer": source_layer,
                "target_layer": target_layer,
                "events": aggregate["events"],
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "specificity": specificity,
                "balanced_accuracy": balanced_accuracy,
                "weighted_recall": weighted_recall,
                "avg_cache_hit_rate": summary["cache_hit_rate_sum"] / summary_count,
                "avg_max_active_experts": summary["max_active_experts_sum"] / summary_count,
                "avg_active_experts": summary["avg_active_experts_sum"] / summary_count,
            }
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--trace_path", type=Path, required=True)
    parser.add_argument("--output_csv", type=Path, required=True)
    parser.add_argument("--output_json", type=Path)
    parser.add_argument("--request_metrics_path", type=Path)
    parser.add_argument("--boundary_window", type=int, default=0)
    args = parser.parse_args()

    request_metadata = load_request_metadata(args.request_metrics_path)
    results = aggregate_confusion(args.trace_path, request_metadata, args.boundary_window)
    if not results:
        raise RuntimeError(f"No confusion rows were found in {args.trace_path}")

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w", encoding="utf-8") as json_file:
            json.dump(results, json_file, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
