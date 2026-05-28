"""Compute dataset counts needed by the LKYWDetection paper placeholders."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

IMG_EXTS = {".bmp", ".dng", ".jpeg", ".jpg", ".mpo", ".png", ".tif", ".tiff", ".webp", ".pfm", ".heic"}


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_root(data_yaml: Path, data: dict[str, Any]) -> Path:
    root = Path(data.get("path") or data_yaml.parent)
    return root if root.is_absolute() else (data_yaml.parent / root).resolve()


def list_images(root: Path, split_value: str) -> list[Path]:
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
        return images
    if split_path.is_dir():
        return sorted(p for p in split_path.rglob("*") if p.suffix.lower() in IMG_EXTS)
    return []


def label_path_for_image(image_path: Path) -> Path:
    parts = list(image_path.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i].lower() == "images":
            parts[i] = "labels"
            return Path(*parts).with_suffix(".txt")
    return image_path.with_suffix(".txt")


def parse_ids(value: str) -> set[int]:
    return {int(x) for x in value.replace(" ", "").split(",") if x != ""}


def count_labels(images: list[Path], fire_ids: set[int], small_area: float) -> dict[str, Any]:
    class_counts: dict[int, int] = {}
    total_boxes = 0
    fire_boxes = 0
    small_fire_boxes = 0
    for image in images:
        label = label_path_for_image(image)
        if not label.exists():
            continue
        for line in label.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls = int(float(parts[0]))
            bw = float(parts[3])
            bh = float(parts[4])
            total_boxes += 1
            class_counts[cls] = class_counts.get(cls, 0) + 1
            if cls in fire_ids:
                fire_boxes += 1
                if bw * bh <= small_area:
                    small_fire_boxes += 1
    return {
        "boxes": total_boxes,
        "class_counts": {str(k): v for k, v in sorted(class_counts.items())},
        "fire_boxes": fire_boxes,
        "small_fire_boxes": small_fire_boxes,
        "small_fire_box_percent": 100.0 * small_fire_boxes / max(fire_boxes, 1),
    }


def optional_subset_percent(root: Path, total_images: int, names: list[str]) -> dict[str, float | int | None]:
    out: dict[str, float | int | None] = {}
    for name in names:
        subset_dir = root / "images" / name
        if subset_dir.is_dir():
            n = len([p for p in subset_dir.rglob("*") if p.suffix.lower() in IMG_EXTS])
            out[f"{name}_images"] = n
            out[f"{name}_image_percent"] = 100.0 * n / max(total_images, 1)
        else:
            out[f"{name}_images"] = None
            out[f"{name}_image_percent"] = None
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--split", default="test")
    parser.add_argument("--fire-class-ids", default="0,2")
    parser.add_argument("--small-area", default=0.01, type=float, help="Normalized xywh area threshold for tiny fire.")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    data = load_yaml(args.data)
    root = resolve_root(args.data, data)
    images = list_images(root, data.get(args.split) or data.get("val"))
    stats = {
        "data": str(args.data),
        "root": str(root),
        "split": args.split,
        "images": len(images),
    }
    stats.update(count_labels(images, parse_ids(args.fire_class_ids), args.small_area))
    stats.update(optional_subset_percent(root, len(images), ["smoke", "small_fire", "night"]))

    text = json.dumps(stats, ensure_ascii=False, indent=2)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
