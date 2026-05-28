"""Validate LKYWDetection and create train/val/test txt splits plus a no-AI train split."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

IMG_EXTS = {".bmp", ".jpeg", ".jpg", ".png", ".webp", ".tif", ".tiff"}
EXPECTED_NAMES = ["carFire", "carNofire", "lkywFire", "lkywNofire"]
REPO = Path(__file__).resolve().parents[1]


def image_files(path: Path) -> list[Path]:
    return sorted(p for p in path.rglob("*") if p.suffix.lower() in IMG_EXTS)


def label_for(image: Path, dataset: Path) -> Path:
    rel = image.relative_to(dataset / "images")
    return dataset / "labels" / rel.with_suffix(".txt")


def read_ai_list(path: Path) -> set[str]:
    if not path.exists():
        return set()
    out = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        item = line.strip()
        if not item or item.startswith("#"):
            continue
        p = Path(item)
        out.add(p.name.lower())
        out.add(p.stem.lower())
    return out


def read_classes(dataset: Path) -> list[str]:
    for split in ("train", "val", "test"):
        path = dataset / "labels" / split / "classes.txt"
        if path.exists():
            return [x.strip() for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    return []


def inspect_split(dataset: Path, split: str) -> dict[str, Any]:
    images = image_files(dataset / "images" / split)
    missing_labels, empty_labels, bad_labels = [], [], []
    class_counts: dict[int, int] = {}
    box_count = 0
    for image in images:
        label = label_for(image, dataset)
        if not label.exists():
            missing_labels.append(image.name)
            continue
        lines = [x.strip() for x in label.read_text(encoding="utf-8", errors="ignore").splitlines() if x.strip()]
        if not lines:
            empty_labels.append(label.name)
            continue
        for line in lines:
            parts = line.split()
            try:
                cls = int(float(parts[0]))
                xywh = [float(x) for x in parts[1:5]]
                if cls < 0 or cls >= len(EXPECTED_NAMES) or len(xywh) != 4 or any(x < 0 or x > 1 for x in xywh):
                    raise ValueError
            except Exception:
                bad_labels.append({"file": label.name, "line": line})
                continue
            class_counts[cls] = class_counts.get(cls, 0) + 1
            box_count += 1

    label_files = [p for p in (dataset / "labels" / split).rglob("*.txt") if p.name != "classes.txt"]
    image_stems = {p.stem for p in images}
    label_stems = {p.stem for p in label_files}
    orphan_labels = sorted(label_stems - image_stems)
    return {
        "images": len(images),
        "labels": len(label_files),
        "boxes": box_count,
        "class_counts": {str(k): v for k, v in sorted(class_counts.items())},
        "missing_labels": missing_labels,
        "orphan_labels": orphan_labels,
        "empty_labels": empty_labels,
        "bad_labels": bad_labels,
        "image_paths": images,
    }


def write_split(path: Path, images: list[Path]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"./{Path(os.path.relpath(p, path.parent)).as_posix()}" for p in images]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def display_path(path: Path) -> str:
    try:
        return Path(os.path.relpath(path, REPO)).as_posix()
    except ValueError:
        return path.as_posix()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=REPO / "datasets" / "LKYWDetection", type=Path)
    parser.add_argument("--ai-list", default="lkyw_experiments/manifests/ai_images.txt", type=Path)
    parser.add_argument("--out-dir", type=Path, help="Defaults to lkyw_experiments/splits.")
    parser.add_argument("--require-ai", action="store_true", help="Fail if the AI manifest is empty.")
    args = parser.parse_args()

    dataset = args.dataset.resolve()
    out_dir = args.out_dir or REPO / "lkyw_experiments" / "splits"
    ai_list = args.ai_list if args.ai_list.is_absolute() else Path.cwd() / args.ai_list
    ai_names = read_ai_list(ai_list)
    if args.require_ai and not ai_names:
        raise SystemExit(f"AI manifest is empty: {ai_list}")

    classes = read_classes(dataset)
    if classes and classes != EXPECTED_NAMES:
        raise SystemExit(f"Unexpected class order {classes}; expected {EXPECTED_NAMES}")

    stats: dict[str, Any] = {"dataset": display_path(dataset), "classes": classes, "splits": {}}
    errors = []
    all_images: dict[str, tuple[str, Path]] = {}
    for split in ("train", "val", "test"):
        split_stats = inspect_split(dataset, split)
        images = split_stats.pop("image_paths")
        stats["splits"][split] = split_stats
        write_split(out_dir / f"{split}.txt", images)
        for image in images:
            all_images[image.name.lower()] = (split, image)
            all_images[image.stem.lower()] = (split, image)
        for key in ("missing_labels", "orphan_labels", "empty_labels", "bad_labels"):
            if split_stats[key]:
                errors.append({split: {key: split_stats[key]}})

    ai_images = []
    unknown_ai = sorted(x for x in ai_names if x not in all_images)
    if unknown_ai:
        errors.append({"ai_manifest_unknown": unknown_ai})
    for name in ai_names:
        if name not in all_images:
            continue
        split, image = all_images[name]
        if split != "train":
            errors.append({"ai_not_in_train": image.as_posix()})
        elif image not in ai_images:
            ai_images.append(image)

    train_images = image_files(dataset / "images" / "train")
    ai_set = {p.resolve() for p in ai_images}
    train_noai = [p for p in train_images if p.resolve() not in ai_set]
    write_split(out_dir / "train_noai.txt", train_noai)

    stats["ai_train_images"] = len(ai_images)
    stats["train_noai_images"] = len(train_noai)
    stats["total_images"] = sum(stats["splits"][s]["images"] for s in ("train", "val", "test"))
    stats["total_boxes"] = sum(stats["splits"][s]["boxes"] for s in ("train", "val", "test"))
    stats["errors"] = errors
    (out_dir / "lkyw_detection_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({k: v for k, v in stats.items() if k != "errors"}, ensure_ascii=False, indent=2))
    if errors:
        print(json.dumps({"errors": errors}, ensure_ascii=False, indent=2))
        raise SystemExit(1)
    if not ai_images:
        print(f"WARNING: no AI images were resolved from {ai_list}; train_noai.txt currently equals train.txt.")
    print(f"Wrote splits and stats to {out_dir}")


if __name__ == "__main__":
    main()
