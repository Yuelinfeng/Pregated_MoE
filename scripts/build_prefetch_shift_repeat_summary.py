import argparse
from pathlib import Path
from typing import Iterable, List

import pandas as pd


def ci95_half_width(series: pd.Series) -> float:
    count = int(series.count())
    if count <= 1:
        return 0.0
    return 1.96 * float(series.std(ddof=1)) / (count ** 0.5)


def iter_seed_dirs(results_root: Path) -> Iterable[Path]:
    for seed_dir in sorted(results_root.glob("seed*")):
        if seed_dir.is_dir():
            yield seed_dir


def load_seeded_csv(seed_dirs: Iterable[Path], relative_path: Path) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for seed_dir in seed_dirs:
        csv_path = seed_dir / relative_path
        if not csv_path.is_file():
            continue
        frame = pd.read_csv(csv_path)
        frame["seed"] = seed_dir.name.replace("seed", "", 1)
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def summarize_numeric(frame: pd.DataFrame, group_columns: List[str], value_columns: List[str]) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()

    rows = []
    for group_key, group in frame.groupby(group_columns):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        row = dict(zip(group_columns, group_key))
        row["num_repeats"] = int(group["seed"].nunique()) if "seed" in group.columns else int(len(group))
        for column in value_columns:
            numeric = pd.to_numeric(group[column], errors="coerce").dropna()
            if numeric.empty:
                continue
            row[f"{column}_mean"] = float(numeric.mean())
            row[f"{column}_std"] = float(numeric.std(ddof=1)) if len(numeric) > 1 else 0.0
            row[f"{column}_ci95"] = ci95_half_width(numeric)
        rows.append(row)
    return pd.DataFrame(rows)


def write_frame(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--results_root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()

    seed_dirs = list(iter_seed_dirs(args.results_root))
    if not seed_dirs:
        raise RuntimeError(f"No seed directories were found under {args.results_root}")

    latency = load_seeded_csv(seed_dirs, Path("latency_compare.csv"))
    if not latency.empty:
        latency_summary = summarize_numeric(
            latency,
            ["condition", "cache_ratio"],
            [
                "on_demand_mean_ms",
                "prefetch_mean_ms",
                "prefetch_over_on_demand",
                "on_demand_p95_ms",
                "prefetch_p95_ms",
            ],
        )
        write_frame(latency, args.output_dir / "latency_repeat_raw.csv")
        write_frame(latency_summary, args.output_dir / "latency_repeat_summary.csv")

    mechanisms = load_seeded_csv(seed_dirs, Path("formal_analysis") / "prefetch_mechanism_metrics.csv")
    if not mechanisms.empty:
        mechanism_value_columns = [
            column
            for column in mechanisms.columns
            if column not in {"condition", "cache_ratio", "seed"}
            and pd.api.types.is_numeric_dtype(mechanisms[column])
        ]
        mechanism_summary = summarize_numeric(mechanisms, ["condition", "cache_ratio"], mechanism_value_columns)
        write_frame(mechanisms, args.output_dir / "mechanism_repeat_raw.csv")
        write_frame(mechanism_summary, args.output_dir / "mechanism_repeat_summary.csv")

    interaction = load_seeded_csv(seed_dirs, Path("formal_analysis") / "factorial_interaction_effects.csv")
    if not interaction.empty:
        interaction_summary = summarize_numeric(
            interaction,
            ["cache_ratio"],
            ["interaction_beta3", "stable_homogeneous", "shifted_homogeneous", "stable_mixed", "shifted_mixed"],
        )
        write_frame(interaction, args.output_dir / "interaction_repeat_raw.csv")
        write_frame(interaction_summary, args.output_dir / "interaction_repeat_summary.csv")


if __name__ == "__main__":
    main()
