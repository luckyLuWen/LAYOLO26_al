"""Run LKYWDetection paper experiments from a YAML plan."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
os.environ.setdefault("YOLO_CONFIG_DIR", str(REPO / ".ultralytics_config"))
(REPO / ".ultralytics_config").mkdir(exist_ok=True)

from ultralytics import YOLO  # noqa: E402
from ultralytics.utils import SETTINGS  # noqa: E402

SETTINGS.update(
    {
        "datasets_dir": str(REPO / "datasets"),
        "weights_dir": str(REPO / "weights"),
        "runs_dir": str(REPO / "runs"),
        "sync": False,
        "clearml": False,
        "comet": False,
        "dvc": False,
        "mlflow": False,
        "neptune": False,
        "tensorboard": False,
        "wandb": False,
    }
)


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def render(value: Any, context: dict[str, str]) -> Any:
    if isinstance(value, str):
        return value.format(**context).replace("\\", "/")
    if isinstance(value, list):
        return [render(v, context) for v in value]
    if isinstance(value, dict):
        return {k: render(v, context) for k, v in value.items()}
    return value


def merge_params(plan: dict[str, Any], params: dict[str, Any] | None) -> dict[str, Any]:
    params = deepcopy(params or {})
    merged: dict[str, Any] = {}
    for block in params.pop("inherit", []):
        merged.update(deepcopy(plan.get(block, {})))
    merged.update(params)
    return merged


def check_weight(path: str, label: str) -> None:
    p = Path(path)
    if p.suffix == ".pt" and (p.is_absolute() or "/" in path or "\\" in path) and not p.exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def last_epoch(results_csv: Path) -> int | None:
    if not results_csv.exists():
        return None
    with results_csv.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    value = (rows[-1].get("epoch") or "").strip()
    try:
        return int(float(value))
    except ValueError:
        return None


def is_complete_run(run_dir: Path, epochs: int | None) -> bool:
    if (run_dir / ".lkyw_complete.json").exists():
        return True

    weights_ok = (run_dir / "weights" / "best.pt").exists() and (run_dir / "weights" / "last.pt").exists()
    results_ok = (run_dir / "results.csv").exists()
    final_plots_ok = (run_dir / "results.png").exists() or (run_dir / "confusion_matrix.png").exists()
    if weights_ok and results_ok and final_plots_ok:
        return True

    end_epoch = last_epoch(run_dir / "results.csv")
    return bool(weights_ok and epochs and end_epoch is not None and end_epoch >= epochs)


def write_complete_marker(run_dir: Path, exp_name: str, seed: int, train_args: dict[str, Any]) -> None:
    marker = {
        "experiment": exp_name,
        "seed": seed,
        "epochs": train_args.get("epochs"),
        "completed_at": datetime.now().isoformat(timespec="seconds"),
    }
    (run_dir / ".lkyw_complete.json").write_text(json.dumps(marker, ensure_ascii=False, indent=2), encoding="utf-8")


def iter_jobs(plan: dict[str, Any], args: argparse.Namespace):
    groups = set(args.group or [])
    only = set(args.only or [])
    for exp in plan.get("experiments", []):
        exp_groups = set(exp.get("groups", []))
        if groups and groups.isdisjoint(exp_groups):
            continue
        if only and exp["name"] not in only:
            continue
        seeds = args.seed if args.seed is not None else exp.get("seeds") or plan.get("seeds", [0])
        for seed in seeds:
            yield exp, int(seed)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", default="lkyw_experiments/plans/paper_full.yaml", type=Path)
    parser.add_argument("--group", nargs="*", help="Run experiments that contain any of these groups.")
    parser.add_argument("--only", nargs="*", help="Run only these experiment names.")
    parser.add_argument("--seed", nargs="*", type=int, help="Override plan seeds.")
    parser.add_argument("--workspace", default=os.environ.get("LKYW_ROOT"), help="Override workspace root.")
    parser.add_argument("--project", help="Override output project directory.")
    parser.add_argument("--device", help="Override train device.")
    parser.add_argument("--epochs", type=int, help="Override epochs.")
    parser.add_argument("--batch", type=int, help="Override batch size.")
    parser.add_argument("--workers", type=int, help="Override dataloader workers.")
    parser.add_argument("--exist-ok", action="store_true", help="Allow overwriting existing run names.")
    parser.add_argument("--no-skip-existing", action="store_true", help="Do not skip runs detected as complete.")
    parser.add_argument("--no-resume-existing", action="store_true", help="Do not resume incomplete runs from weights/last.pt.")
    parser.add_argument("--dry-run", action="store_true", help="Print resolved jobs without training.")
    args = parser.parse_args()

    plan_path = args.plan if args.plan.is_absolute() else REPO / args.plan
    plan = load_yaml(plan_path)
    repo_root = REPO.as_posix()
    workspace_template = args.workspace or plan.get("workspace", repo_root)
    workspace = render(workspace_template, {"repo": repo_root})
    context = {"repo": repo_root, "workspace": Path(workspace).as_posix()}
    context["weights_dir"] = render(plan.get("weights_dir", "{workspace}/weights"), context)
    context["project"] = args.project or render(plan.get("project", "{workspace}/runs/lkyw"), context)

    plan = render(plan, context)
    data_aliases = plan.get("data", {})

    selected = list(iter_jobs(plan, args))
    if not selected:
        print("No jobs selected.")
        return

    for exp, seed in selected:
        train_args = deepcopy(plan.get("train_defaults", {}))
        train_args.update(merge_params(plan, exp.get("params")))
        train_args["data"] = data_aliases.get(exp["data"], exp["data"])
        train_args["project"] = context["project"]
        train_args["name"] = f"{exp['name']}_seed{seed}"
        train_args["seed"] = seed
        train_args["exist_ok"] = args.exist_ok
        run_dir = Path(train_args["project"]) / train_args["name"]

        for key in ("device", "epochs", "batch", "workers"):
            value = getattr(args, key)
            if value is not None:
                train_args[key] = value

        model_path = exp["model"]
        pretrained = exp.get("pretrained")
        if pretrained:
            train_args["pretrained"] = pretrained

        complete = is_complete_run(run_dir, train_args.get("epochs")) if run_dir.exists() else False
        resume_path = run_dir / "weights" / "last.pt"

        if args.dry_run:
            status = "new"
            if complete and not args.no_skip_existing and not args.exist_ok:
                status = "complete, will skip"
            elif run_dir.exists() and resume_path.exists() and not args.no_resume_existing and not args.exist_ok:
                status = f"incomplete, will resume from {resume_path.as_posix()}"
            elif run_dir.exists():
                status = "exists"
            print(f"[DRY] {train_args['name']} ({status})")
            print(f"      model={model_path}")
            if pretrained:
                print(f"      pretrained={pretrained}")
            print(f"      data={train_args['data']}")
            continue

        if run_dir.exists() and not args.exist_ok:
            if complete and not args.no_skip_existing:
                print(f"[SKIP] {train_args['name']} already complete: {run_dir}")
                continue
            if resume_path.exists() and not args.no_resume_existing:
                print(f"[RESUME] {train_args['name']} from {resume_path}")
                model_path = resume_path.as_posix()
                train_args["resume"] = True
                train_args.pop("pretrained", None)
            else:
                print(
                    f"[SKIP] {train_args['name']} exists but is not complete and has no weights/last.pt. "
                    "Inspect the run folder or pass --exist-ok intentionally."
                )
                continue

        check_weight(model_path, "model")
        if train_args.get("pretrained"):
            check_weight(str(train_args["pretrained"]), "pretrained")
        YOLO(model_path).train(**train_args)
        write_complete_marker(run_dir, exp["name"], seed, train_args)


if __name__ == "__main__":
    main()
