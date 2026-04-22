import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List


MAIN_CONDITIONS = [
    "stable_homogeneous",
    "shifted_homogeneous",
    "stable_mixed",
    "shifted_mixed",
]

ABLATION_CONDITIONS = [
    "stable_mixed_ab",
    "stable_mixed_ba",
    "stable_mixed_balanced_order",
    "shifted_mixed_ratio_only",
    "shifted_mixed_order_only",
    "shifted_mixed_ratio_and_order",
]

METHODS = ["on_demand", "prefetch"]


def run_command(command: List[str], log_path: Path, cwd: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write("COMMAND:\n")
        log_file.write(" ".join(command))
        log_file.write("\n\n")
        log_file.flush()
        process = subprocess.run(command, cwd=str(cwd), stdout=log_file, stderr=subprocess.STDOUT, text=True)
    if process.returncode != 0:
        tail_lines = []
        with log_path.open("r", encoding="utf-8") as log_file:
            tail_lines = log_file.readlines()[-80:]
        tail_text = "".join(tail_lines).strip()
        raise RuntimeError(
            f"Command failed with exit code {process.returncode}: {' '.join(command)}\n"
            f"--- log tail ({log_path}) ---\n{tail_text}"
        )


def count_nonempty_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def salvage_complete_cell(
    cell_dir: Path,
    *,
    expected_num_requests: int,
    method: str,
    condition: str,
    cache_ratio: str,
    seed: int,
) -> bool:
    request_trace_path = cell_dir / "request_trace.jsonl"
    workload_summary_path = cell_dir / "workload_summary.json"
    request_metrics_path = cell_dir / "request_metrics.jsonl"
    request_metrics_partial_path = cell_dir / "request_metrics.jsonl.partial"
    run_status_path = cell_dir / "run_status.json"

    if request_metrics_partial_path.is_file() and count_nonempty_lines(request_metrics_partial_path) == expected_num_requests:
        if request_metrics_path.exists():
            request_metrics_path.unlink()
        request_metrics_partial_path.replace(request_metrics_path)

    required_inputs_present = request_trace_path.is_file() and workload_summary_path.is_file() and request_metrics_path.is_file()
    if not required_inputs_present:
        return False

    if count_nonempty_lines(request_trace_path) != expected_num_requests:
        return False
    if count_nonempty_lines(request_metrics_path) != expected_num_requests:
        return False

    run_status = {}
    if run_status_path.is_file():
        with run_status_path.open("r", encoding="utf-8") as handle:
            run_status = json.load(handle)

    run_status.update(
        {
            "condition": condition,
            "method": method,
            "cache_ratio": float(cache_ratio),
            "seed": seed,
            "expected_num_requests": expected_num_requests,
            "completed_num_requests": expected_num_requests,
            "status": "complete",
            "completed_at_utc": utc_timestamp(),
            "salvaged_after_nonzero_exit": True,
        }
    )
    if "started_at_utc" not in run_status:
        run_status["started_at_utc"] = utc_timestamp()
    if "trace_id" not in run_status:
        run_status["trace_id"] = f"{condition}_cr{cache_ratio}_{method}_seed{seed}"

    with run_status_path.open("w", encoding="utf-8") as handle:
        json.dump(run_status, handle, ensure_ascii=False, indent=2)

    return cell_complete(cell_dir, expected_num_requests, method)


def cell_complete(cell_dir: Path, expected_num_requests: int, method: str) -> bool:
    run_status_path = cell_dir / "run_status.json"
    request_trace_path = cell_dir / "request_trace.jsonl"
    request_metrics_path = cell_dir / "request_metrics.jsonl"
    workload_summary_path = cell_dir / "workload_summary.json"
    if not all(path.is_file() for path in [run_status_path, request_trace_path, request_metrics_path, workload_summary_path]):
        return False

    run_status = json.load(run_status_path.open("r", encoding="utf-8"))
    if run_status.get("status") != "complete":
        return False
    if int(run_status.get("completed_num_requests", -1)) != expected_num_requests:
        return False
    if count_nonempty_lines(request_trace_path) != expected_num_requests:
        return False
    if count_nonempty_lines(request_metrics_path) != expected_num_requests:
        return False
    if method == "prefetch" and not (cell_dir / "confusion_boundary8.csv").is_file():
        return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--results_root", type=Path, required=True)
    parser.add_argument("--repo_root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--ckpt_path", type=Path, required=True)
    parser.add_argument("--offload_path", type=Path, required=True)
    parser.add_argument("--lib_path", type=str, default="build/lib/libth_transformer.so")
    parser.add_argument("--conditions", nargs="*", default=MAIN_CONDITIONS)
    parser.add_argument("--methods", nargs="*", default=METHODS)
    parser.add_argument("--cache_ratios", nargs="*", default=["0.03", "0.1", "0.4"])
    parser.add_argument("--num_requests", type=int, default=128)
    parser.add_argument("--shift_block_size", type=int, default=32)
    parser.add_argument("--domain_order", type=str, default="translation,summarization")
    parser.add_argument("--homogeneous_word_budget", type=int, default=256)
    parser.add_argument("--mix_word_budget", type=int, default=256)
    parser.add_argument("--stable_mix_fraction", type=float, default=0.5)
    parser.add_argument("--shift_major_fraction", type=float, default=0.8)
    parser.add_argument("--mix_mode", choices=["concat", "interleave"], default="interleave")
    parser.add_argument("--interleave_chunk_words", type=int, default=16)
    parser.add_argument("--beam_width", type=int, default=1)
    parser.add_argument("--max_seq_len", type=int, default=128)
    parser.add_argument("--sampling_topk", type=int, default=1)
    parser.add_argument("--sampling_topp", type=float, default=0.0)
    parser.add_argument("--moe_topk", type=int, default=1)
    parser.add_argument("--data_type", choices=["fp32", "fp16"], default="fp32")
    parser.add_argument("--tensor_para_size", type=int, default=1)
    parser.add_argument("--pipeline_para_size", type=int, default=1)
    parser.add_argument("--cache_policy", choices=["LFU", "LRU", "LIFO"], default="LFU")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_retries", type=int, default=2)
    parser.add_argument("--force_rerun", action="store_true")
    parser.add_argument("--skip_plots", action="store_true")
    args = parser.parse_args()

    python_exe = sys.executable
    repo_root = args.repo_root.resolve()
    args.results_root.mkdir(parents=True, exist_ok=True)
    selected_methods = list(dict.fromkeys(args.methods))
    selected_conditions = list(dict.fromkeys(args.conditions))
    has_on_demand = "on_demand" in selected_methods
    has_prefetch = "prefetch" in selected_methods
    has_comparable_methods = has_on_demand and has_prefetch
    is_main_2x2_run = set(selected_conditions).issubset(set(MAIN_CONDITIONS))
    has_ablation_conditions = any(condition in ABLATION_CONDITIONS for condition in selected_conditions)

    for method in selected_methods:
        for condition in selected_conditions:
            for cache_ratio in args.cache_ratios:
                cell_dir = args.results_root / method / condition / f"cr{cache_ratio}"
                if not args.force_rerun and cell_complete(cell_dir, args.num_requests, method):
                    print(f"[skip] {method} {condition} cr{cache_ratio}")
                    continue

                success = False
                for attempt in range(1, args.max_retries + 1):
                    print(f"[run] {method} {condition} cr{cache_ratio} attempt={attempt}")
                    shutil.rmtree(cell_dir, ignore_errors=True)
                    cell_dir.mkdir(parents=True, exist_ok=True)

                    trace_command = [
                        python_exe,
                        str(repo_root / "scripts" / "eval_prefetch_shift.py"),
                        "--output_dir", str(cell_dir),
                        "--trace_id", f"{condition}_cr{cache_ratio}_{method}_seed{args.seed}",
                        "--condition", condition,
                        "--method", method,
                        "--model_path", args.model_path,
                        "--ckpt_path", str(args.ckpt_path),
                        "--offload_path", str(args.offload_path),
                        "--lib_path", args.lib_path,
                        "--num_requests", str(args.num_requests),
                        "--shift_block_size", str(args.shift_block_size),
                        "--domain_order", args.domain_order,
                        "--match_domain_marginal",
                        "--homogeneous_word_budget", str(args.homogeneous_word_budget),
                        "--mix_word_budget", str(args.mix_word_budget),
                        "--stable_mix_fraction", str(args.stable_mix_fraction),
                        "--shift_major_fraction", str(args.shift_major_fraction),
                        "--mix_mode", args.mix_mode,
                        "--interleave_chunk_words", str(args.interleave_chunk_words),
                        "--beam_width", str(args.beam_width),
                        "--max_seq_len", str(args.max_seq_len),
                        "--sampling_topk", str(args.sampling_topk),
                        "--sampling_topp", str(args.sampling_topp),
                        "--moe_topk", str(args.moe_topk),
                        "--data_type", args.data_type,
                        "--tensor_para_size", str(args.tensor_para_size),
                        "--pipeline_para_size", str(args.pipeline_para_size),
                        "--cache_ratio", str(cache_ratio),
                        "--cache_policy", args.cache_policy,
                        "--seed", str(args.seed),
                    ]

                    trace_failed = False
                    try:
                        run_command(trace_command, cell_dir / f"run_attempt{attempt}.log", repo_root)
                    except Exception as exc:
                        trace_failed = True
                        salvaged = salvage_complete_cell(
                            cell_dir,
                            expected_num_requests=args.num_requests,
                            method=method,
                            condition=condition,
                            cache_ratio=str(cache_ratio),
                            seed=args.seed,
                        )
                        if salvaged:
                            print(
                                f"[salvaged] completed outputs recovered after nonzero exit for "
                                f"{method} {condition} cr{cache_ratio}"
                            )
                        else:
                            print(f"[warn] attempt {attempt} failed for {method} {condition} cr{cache_ratio}: {exc}")
                            continue

                    if method == "prefetch" and not (cell_dir / "confusion_boundary8.csv").is_file():
                        analyze_command = [
                            python_exe,
                            str(repo_root / "scripts" / "analyze_prefetch_confusion.py"),
                            "--trace_path", str(cell_dir / "prefetch_trace.tsv"),
                            "--request_metrics_path", str(cell_dir / "request_metrics.jsonl"),
                            "--boundary_window", "8",
                            "--output_csv", str(cell_dir / "confusion_boundary8.csv"),
                            "--output_json", str(cell_dir / "confusion_boundary8.json"),
                        ]
                        try:
                            run_command(analyze_command, cell_dir / f"analyze_attempt{attempt}.log", repo_root)
                        except Exception as exc:
                            print(
                                f"[warn] confusion analysis failed for {method} {condition} cr{cache_ratio}: {exc}"
                            )
                            continue

                    if cell_complete(cell_dir, args.num_requests, method):
                        success = True
                        break

                if not success:
                    raise RuntimeError(f"Failed to complete cell {method} {condition} cr{cache_ratio} after retries.")

    validate_command = [
        python_exe,
        str(repo_root / "scripts" / "validate_prefetch_shift_results.py"),
        "--results_root", str(args.results_root),
        "--expected_num_requests", str(args.num_requests),
        "--conditions", *selected_conditions,
        "--methods", *args.methods,
        "--cache_ratios", *args.cache_ratios,
    ]
    run_command(validate_command, args.results_root / "validate.log", repo_root)

    if not has_comparable_methods:
        print(
            "[skip] build_prefetch_shift_latency_compare.py requires both "
            "'on_demand' and 'prefetch' methods."
        )
        print(
            "[skip] analyze_prefetch_mechanisms.py and plotting also require "
            "prefetch/on-demand comparison outputs."
        )
        print(f"Completed formal run at {args.results_root}")
        return

    latency_command = [
        python_exe,
        str(repo_root / "scripts" / "build_prefetch_shift_latency_compare.py"),
        "--results_root", str(args.results_root),
        "--expected_num_requests", str(args.num_requests),
        "--conditions", *selected_conditions,
        "--cache_ratios", *args.cache_ratios,
    ]
    run_command(latency_command, args.results_root / "build_latency_compare.log", repo_root)

    mechanism_command = [
        python_exe,
        str(repo_root / "scripts" / "analyze_prefetch_mechanisms.py"),
        "--results_root", str(args.results_root),
        "--output_dir", str(args.results_root / "formal_analysis"),
        "--expert_bytes", "18874368",
    ]
    run_command(mechanism_command, args.results_root / "formal_analysis.log", repo_root)

    if has_ablation_conditions:
        ablation_command = [
            python_exe,
            str(repo_root / "scripts" / "analyze_ratio_order_ablation.py"),
            "--results_root", str(args.results_root),
        ]
        run_command(ablation_command, args.results_root / "ablation_analysis.log", repo_root)

    if not args.skip_plots:
        if is_main_2x2_run:
            plot_final_command = [
                python_exe,
                str(repo_root / "scripts" / "plot_prefetch_shift_final_figure.py"),
                "--results_root", str(args.results_root),
                "--output_dir", str(args.results_root / "figures"),
            ]
            plot_results_command = [
                python_exe,
                str(repo_root / "scripts" / "plot_prefetch_shift_results.py"),
                "--results_root", str(args.results_root),
                "--output_dir", str(args.results_root / "figures"),
            ]
            run_command(plot_final_command, args.results_root / "plot_final.log", repo_root)
            run_command(plot_results_command, args.results_root / "plot_results.log", repo_root)
        else:
            if has_ablation_conditions:
                ablation_plot_command = [
                    python_exe,
                    str(repo_root / "scripts" / "plot_ratio_order_ablation_summary.py"),
                    "--results_root", str(args.results_root),
                    "--output_dir", str(args.results_root / "figures"),
                ]
                run_command(ablation_plot_command, args.results_root / "plot_ablation.log", repo_root)
            print(
                "[skip] main 2x2 plotting is only defined for "
                f"{MAIN_CONDITIONS}; current conditions are {selected_conditions}."
            )
            if has_ablation_conditions:
                print(
                    "[done] ratio/order ablation summary CSVs and single-figure plot are in "
                    f"{args.results_root / 'ablation_analysis'} and {args.results_root / 'figures'}."
                )

    print(f"Completed formal run at {args.results_root}")


if __name__ == "__main__":
    main()
