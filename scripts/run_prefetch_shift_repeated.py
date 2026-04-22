import argparse
import subprocess
import sys
from pathlib import Path
from typing import List


def run_command(command: List[str], cwd: Path) -> None:
    print("[run]", " ".join(command))
    process = subprocess.run(command, cwd=str(cwd), text=True)
    if process.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {process.returncode}: {' '.join(command)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            "Run the prefetch-shift formal suite for multiple seeds and build repeat summaries. "
            "All arguments other than the ones defined here are forwarded to run_prefetch_shift_formal.py."
        ),
    )
    parser.add_argument("--results_root", type=Path, required=True)
    parser.add_argument("--repo_root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--skip_summary", action="store_true")
    parser.add_argument("--summary_output_dir", type=Path)
    args, passthrough = parser.parse_known_args()

    repo_root = args.repo_root.resolve()
    python_exe = sys.executable

    for seed in args.seeds:
        seed_results_root = args.results_root / f"seed{seed}"
        command = [
            python_exe,
            str(repo_root / "scripts" / "run_prefetch_shift_formal.py"),
            "--results_root",
            str(seed_results_root),
            "--seed",
            str(seed),
            *passthrough,
        ]
        run_command(command, repo_root)

    if args.skip_summary:
        return

    summary_output_dir = args.summary_output_dir or (args.results_root / "repeat_summary")
    summary_command = [
        python_exe,
        str(repo_root / "scripts" / "build_prefetch_shift_repeat_summary.py"),
        "--results_root",
        str(args.results_root),
        "--output_dir",
        str(summary_output_dir),
    ]
    run_command(summary_command, repo_root)


if __name__ == "__main__":
    main()
