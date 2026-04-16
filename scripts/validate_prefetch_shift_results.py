import argparse
import json
from pathlib import Path
from typing import Dict, List


MAIN_CONDITIONS = [
    "stable_homogeneous",
    "shifted_homogeneous",
    "stable_mixed",
    "shifted_mixed",
]

METHODS = ["on_demand", "prefetch"]


def count_nonempty_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def load_json(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def validate_cell(
    cell_dir: Path,
    *,
    expected_num_requests: int,
    method: str,
    require_confusion: bool,
) -> List[str]:
    errors: List[str] = []
    request_trace_path = cell_dir / "request_trace.jsonl"
    workload_summary_path = cell_dir / "workload_summary.json"
    request_metrics_path = cell_dir / "request_metrics.jsonl"
    request_metrics_partial_path = cell_dir / "request_metrics.jsonl.partial"
    run_status_path = cell_dir / "run_status.json"

    for required_path in [request_trace_path, workload_summary_path, request_metrics_path, run_status_path]:
        if not required_path.is_file():
            errors.append(f"missing required file: {required_path}")

    if request_metrics_partial_path.exists():
        errors.append(f"partial request metrics file still present: {request_metrics_partial_path}")

    if require_confusion:
        confusion_csv_path = cell_dir / "confusion_boundary8.csv"
        if not confusion_csv_path.is_file():
            errors.append(f"missing required file: {confusion_csv_path}")

    if errors:
        return errors

    request_trace_lines = count_nonempty_lines(request_trace_path)
    request_metrics_lines = count_nonempty_lines(request_metrics_path)
    workload_summary = load_json(workload_summary_path)
    run_status = load_json(run_status_path)

    if request_trace_lines != expected_num_requests:
        errors.append(
            f"{cell_dir}: request_trace.jsonl has {request_trace_lines} rows, expected {expected_num_requests}"
        )
    if request_metrics_lines != expected_num_requests:
        errors.append(
            f"{cell_dir}: request_metrics.jsonl has {request_metrics_lines} rows, expected {expected_num_requests}"
        )
    if int(workload_summary.get("num_requests", -1)) != expected_num_requests:
        errors.append(
            f"{cell_dir}: workload_summary num_requests={workload_summary.get('num_requests')} "
            f"expected {expected_num_requests}"
        )
    if int(run_status.get("expected_num_requests", -1)) != expected_num_requests:
        errors.append(
            f"{cell_dir}: run_status expected_num_requests={run_status.get('expected_num_requests')} "
            f"expected {expected_num_requests}"
        )
    if int(run_status.get("completed_num_requests", -1)) != expected_num_requests:
        errors.append(
            f"{cell_dir}: run_status completed_num_requests={run_status.get('completed_num_requests')} "
            f"expected {expected_num_requests}"
        )
    if run_status.get("status") not in {"complete", "trace_only_complete"}:
        errors.append(f"{cell_dir}: run_status status={run_status.get('status')} is not complete")
    if str(run_status.get("method")) != method:
        errors.append(f"{cell_dir}: run_status method={run_status.get('method')} expected {method}")

    return errors


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--results_root", type=Path, required=True)
    parser.add_argument("--conditions", nargs="*", default=MAIN_CONDITIONS)
    parser.add_argument("--methods", nargs="*", default=METHODS)
    parser.add_argument("--cache_ratios", nargs="*", default=["0.03", "0.1", "0.4"])
    parser.add_argument("--expected_num_requests", type=int, default=128)
    args = parser.parse_args()

    errors: List[str] = []
    for method in args.methods:
        for condition in args.conditions:
            for cache_ratio in args.cache_ratios:
                cell_dir = args.results_root / method / condition / f"cr{cache_ratio}"
                if not cell_dir.is_dir():
                    errors.append(f"missing cell directory: {cell_dir}")
                    continue
                errors.extend(
                    validate_cell(
                        cell_dir,
                        expected_num_requests=args.expected_num_requests,
                        method=method,
                        require_confusion=(method == "prefetch"),
                    )
                )

    if errors:
        for error in errors:
            print(f"[ERROR] {error}")
        raise SystemExit(1)

    print("All cells are complete and consistent.")


if __name__ == "__main__":
    main()
