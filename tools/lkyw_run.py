"""Run LKYWDetection paper experiments from a YAML plan."""

from __future__ import annotations

import argparse
import os
import sys
from copy import deepcopy
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

        for key in ("device", "epochs", "batch", "workers"):
            value = getattr(args, key)
            if value is not None:
                train_args[key] = value

        model_path = exp["model"]
        pretrained = exp.get("pretrained")
        if pretrained:
            train_args["pretrained"] = pretrained

        if args.dry_run:
            print(f"[DRY] {train_args['name']}")
            print(f"      model={model_path}")
            if pretrained:
                print(f"      pretrained={pretrained}")
            print(f"      data={train_args['data']}")
            continue

        check_weight(model_path, "model")
        if pretrained:
            check_weight(pretrained, "pretrained")
        YOLO(model_path).train(**train_args)


if __name__ == "__main__":
    main()
