import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd


MAIN_CONDITIONS = [
    "stable_homogeneous",
    "shifted_homogeneous",
    "stable_mixed",
    "shifted_mixed",
]


def load_request_metrics(path: Path) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle):
            if not line.strip():
                continue
            payload = json.loads(line)
            rows.append(
                {
                    "request_index": line_number,
                    "request_id": payload["request_id"],
                    "condition": payload["condition"],
                    "domain_label": payload["domain_label"],
                    "latency_ms": float(payload["latency_s"]) * 1000.0,
                    "input_token_count": int(payload["input_token_count"]),
                    "output_token_count": int(payload["output_token_count"]),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--results_root", type=Path, required=True)
    parser.add_argument("--output_csv", type=Path)
    parser.add_argument("--conditions", nargs="*", default=MAIN_CONDITIONS)
    parser.add_argument("--cache_ratios", nargs="*", default=["0.03", "0.1", "0.4"])
    parser.add_argument("--expected_num_requests", type=int, default=128)
    args = parser.parse_args()

    output_csv = args.output_csv or (args.results_root / "latency_compare.csv")
    rows: List[Dict[str, object]] = []

    for condition in args.conditions:
        for cache_ratio in args.cache_ratios:
            on_demand_path = args.results_root / "on_demand" / condition / f"cr{cache_ratio}" / "request_metrics.jsonl"
            prefetch_path = args.results_root / "prefetch" / condition / f"cr{cache_ratio}" / "request_metrics.jsonl"
            on_demand = load_request_metrics(on_demand_path)
            prefetch = load_request_metrics(prefetch_path)

            if len(on_demand) != args.expected_num_requests or len(prefetch) != args.expected_num_requests:
                raise RuntimeError(
                    f"Incomplete cell {condition} cr{cache_ratio}: "
                    f"on_demand={len(on_demand)}, prefetch={len(prefetch)}, expected={args.expected_num_requests}"
                )

            merged = on_demand.merge(
                prefetch,
                on="request_index",
                suffixes=("_on_demand", "_prefetch"),
                how="inner",
            )

            if len(merged) != args.expected_num_requests:
                raise RuntimeError(
                    f"Failed to align request metrics for {condition} cr{cache_ratio}: "
                    f"merged={len(merged)}, expected={args.expected_num_requests}"
                )

            rows.append(
                {
                    "condition": condition,
                    "cache_ratio": float(cache_ratio),
                    "on_demand_mean_ms": merged["latency_ms_on_demand"].mean(),
                    "prefetch_mean_ms": merged["latency_ms_prefetch"].mean(),
                    "prefetch_over_on_demand": merged["latency_ms_prefetch"].mean() / merged["latency_ms_on_demand"].mean(),
                    "on_demand_p95_ms": merged["latency_ms_on_demand"].quantile(0.95),
                    "prefetch_p95_ms": merged["latency_ms_prefetch"].quantile(0.95),
                }
            )

    frame = pd.DataFrame(rows).sort_values(["condition", "cache_ratio"]).reset_index(drop=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_csv, index=False)
    print(output_csv)
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
