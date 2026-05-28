"""Validate a trained model and compute fire/no-fire state metrics for paper tables."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
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

IMG_EXTS = {".bmp", ".dng", ".jpeg", ".jpg", ".mpo", ".png", ".tif", ".tiff", ".webp", ".pfm", ".heic"}


def parse_ids(value: str) -> set[int]:
    return {int(x) for x in value.replace(" ", "").split(",") if x != ""}


def load_data_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    data["yaml_file"] = str(path)
    return data


def resolve_split(data_yaml: Path, split: str) -> tuple[list[Path], str]:
    data = load_data_yaml(data_yaml)
    root = Path(data.get("path") or data_yaml.parent)
    if not root.is_absolute():
        root = (data_yaml.parent / root).resolve()
    split_value = data.get(split) or data.get("val")
    split_path = Path(split_value)
    if not split_path.is_absolute():
        split_path = root / split_path
    if split_path.is_file() and split_path.suffix == ".txt":
        images = []
        for line in split_path.read_text(encoding="utf-8").splitlines():
            item = line.strip()
            if not item:
                continue
            p = Path(item)
            images.append(p if p.is_absolute() else (split_path.parent / p).resolve())
        return images, str(split_path)
    if split_path.is_dir():
        images = [p for p in split_path.rglob("*") if p.suffix.lower() in IMG_EXTS]
        return sorted(images), str(split_path)
    return [split_path], str(split_path)


def label_path_for_image(image_path: Path) -> Path:
    parts = list(image_path.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i].lower() == "images":
            parts[i] = "labels"
            return Path(*parts).with_suffix(".txt")
    return image_path.with_suffix(".txt")


def load_yolo_labels(label_path: Path, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    h, w = shape
    boxes, classes = [], []
    if not label_path.exists():
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls, x, y, bw, bh = map(float, parts[:5])
        x1 = (x - bw / 2) * w
        y1 = (y - bh / 2) * h
        x2 = (x + bw / 2) * w
        y2 = (y + bh / 2) * h
        boxes.append([x1, y1, x2, y2])
        classes.append(int(cls))
    return np.asarray(boxes, dtype=np.float32), np.asarray(classes, dtype=np.int64)


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    area_a = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    return inter / np.clip(area_a[:, None] + area_b[None, :] - inter, 1e-9, None)


def compute_state_metrics(model: YOLO, source: str, images: list[Path], args: argparse.Namespace) -> dict[str, float | int]:
    fire_ids = parse_ids(args.fire_class_ids)
    nofire_ids = parse_ids(args.nofire_class_ids)
    gt_total = gt_fire = gt_nofire = 0
    matched_fire = matched_nofire = state_confusions = nofire_as_fire = fire_as_nofire = 0

    image_set = {str(p.resolve()).lower() for p in images}
    results = model.predict(
        source=source,
        stream=True,
        conf=args.conf,
        iou=args.nms_iou,
        imgsz=args.imgsz,
        device=args.device,
        save=False,
        verbose=False,
    )
    for result in results:
        image_path = Path(result.path).resolve()
        if image_set and str(image_path).lower() not in image_set:
            continue
        gt_boxes, gt_cls = load_yolo_labels(label_path_for_image(image_path), result.orig_shape)
        gt_total += len(gt_cls)
        gt_fire += int(np.isin(gt_cls, list(fire_ids)).sum())
        gt_nofire += int(np.isin(gt_cls, list(nofire_ids)).sum())
        if result.boxes is None or len(result.boxes) == 0 or len(gt_boxes) == 0:
            continue

        pred_boxes = result.boxes.xyxy.cpu().numpy().astype(np.float32)
        pred_cls = result.boxes.cls.cpu().numpy().astype(np.int64)
        pred_conf = result.boxes.conf.cpu().numpy()
        order = np.argsort(-pred_conf)
        ious = iou_matrix(pred_boxes, gt_boxes)
        used_gt: set[int] = set()
        for pred_i in order:
            candidates = [(gt_i, ious[pred_i, gt_i]) for gt_i in range(len(gt_boxes)) if gt_i not in used_gt]
            if not candidates:
                continue
            gt_i, best_iou = max(candidates, key=lambda x: x[1])
            if best_iou < args.match_iou:
                continue
            used_gt.add(gt_i)
            pred_fire = int(pred_cls[pred_i]) in fire_ids
            gt_is_fire = int(gt_cls[gt_i]) in fire_ids
            gt_is_nofire = int(gt_cls[gt_i]) in nofire_ids
            if gt_is_fire and pred_fire:
                matched_fire += 1
            if gt_is_nofire and not pred_fire:
                matched_nofire += 1
            if gt_is_fire != pred_fire and (gt_is_fire or gt_is_nofire):
                state_confusions += 1
                if gt_is_nofire and pred_fire:
                    nofire_as_fire += 1
                if gt_is_fire and not pred_fire:
                    fire_as_nofire += 1

    denom_state = max(gt_fire + gt_nofire, 1)
    return {
        "gt_total": gt_total,
        "gt_fire": gt_fire,
        "gt_nofire": gt_nofire,
        "fire_recall": matched_fire / max(gt_fire, 1),
        "nofire_recall": matched_nofire / max(gt_nofire, 1),
        "fire_nofire_confusion_rate": state_confusions / denom_state,
        "nofire_as_fire_rate": nofire_as_fire / max(gt_nofire, 1),
        "fire_as_nofire_rate": fire_as_nofire / max(gt_fire, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--split", default="test")
    parser.add_argument("--imgsz", default=640, type=int)
    parser.add_argument("--device", default="0")
    parser.add_argument("--conf", default=0.001, type=float)
    parser.add_argument("--nms-iou", default=0.7, type=float)
    parser.add_argument("--match-iou", default=0.5, type=float)
    parser.add_argument("--fire-class-ids", default="0,2")
    parser.add_argument("--nofire-class-ids", default="1,3")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--skip-val", action="store_true")
    args = parser.parse_args()

    model = YOLO(args.model)
    images, source = resolve_split(args.data, args.split)
    output: dict[str, Any] = {
        "model": args.model,
        "data": str(args.data),
        "split": args.split,
        "images": len(images),
    }

    if not args.skip_val:
        metrics = model.val(data=str(args.data), split=args.split, imgsz=args.imgsz, device=args.device, plots=True)
        output.update(
            {
                "map50": float(metrics.box.map50),
                "map50_95": float(metrics.box.map),
                "per_class_ap50_95": [float(x) for x in getattr(metrics.box, "maps", [])],
            }
        )

    output.update(compute_state_metrics(model, source, images, args))
    out_path = args.out or Path(os.environ.get("LKYW_RUNS", REPO / "runs" / "lkyw")) / "validation_metrics.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
