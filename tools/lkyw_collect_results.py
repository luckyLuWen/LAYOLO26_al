"""Collect YOLO training result CSVs into per-run and mean/std summaries."""

from __future__ import annotations

import argparse
import csv
import os
import re
from pathlib import Path
from statistics import mean, pstdev

REPO = Path(__file__).resolve().parents[1]


METRIC_COLUMNS = {
    "precision": "metrics/precision(B)",
    "recall": "metrics/recall(B)",
    "map50": "metrics/mAP50(B)",
    "map50_95": "metrics/mAP50-95(B)",
}


def parse_float(row: dict[str, str], key: str) -> float | None:
    value = row.get(key, "").strip()
    if value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def best_row(csv_path: Path) -> dict[str, str] | None:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    key = METRIC_COLUMNS["map50_95"]
    return max(rows, key=lambda r: parse_float(r, key) if parse_float(r, key) is not None else -1.0)


def base_name(run_name: str) -> str:
    return re.sub(r"_seed\d+$", "", run_name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", default=os.environ.get("LKYW_RUNS", REPO / "runs" / "lkyw"), type=Path)
    parser.add_argument("--out-dir", type=Path, help="Output directory. Defaults to --runs.")
    args = parser.parse_args()

    runs = args.runs
    out_dir = args.out_dir or runs
    out_dir.mkdir(parents=True, exist_ok=True)

    per_runs = []
    for csv_path in sorted(runs.glob("*/results.csv")):
        row = best_row(csv_path)
        if not row:
            continue
        record = {
            "run": csv_path.parent.name,
            "experiment": base_name(csv_path.parent.name),
            "best_epoch": row.get("epoch", ""),
        }
        for name, col in METRIC_COLUMNS.items():
            value = parse_float(row, col)
            record[name] = "" if value is None else f"{value:.6f}"
        per_runs.append(record)

    per_run_path = out_dir / "lkyw_per_run_summary.csv"
    summary_path = out_dir / "lkyw_mean_std_summary.csv"
    fieldnames = ["run", "experiment", "best_epoch", *METRIC_COLUMNS.keys()]
    with per_run_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(per_runs)

    grouped: dict[str, list[dict[str, str]]] = {}
    for row in per_runs:
        grouped.setdefault(row["experiment"], []).append(row)

    summary_fields = ["experiment", "n"]
    for key in METRIC_COLUMNS:
        summary_fields.extend([f"{key}_mean", f"{key}_std"])
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        for exp, rows in sorted(grouped.items()):
            out = {"experiment": exp, "n": len(rows)}
            for key in METRIC_COLUMNS:
                values = [float(r[key]) for r in rows if r.get(key) not in ("", None)]
                out[f"{key}_mean"] = "" if not values else f"{mean(values):.6f}"
                out[f"{key}_std"] = "" if len(values) < 2 else f"{pstdev(values):.6f}"
            writer.writerow(out)

    print(f"Wrote {per_run_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
