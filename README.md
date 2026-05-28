# LA-YOLO26 Portable Experiment Package

This folder is a trimmed and portable Ultralytics-based training package for the LKYWDetection paper experiments. It keeps the modified model code, experiment plans, dataset YAML files, helper scripts, datasets, and pretrained weights needed to reproduce the LA-YOLO26 runs.

## Package Layout

```text
LAYOLO26/
  ultralytics/                 # modified Ultralytics package
    nn/modules/block.py        # SFGA and C3k2SFGA
    nn/modules/__init__.py     # module exports
    nn/tasks.py                # YAML parser support for new modules
    utils/loss.py              # focal modulation, fire-state loss, ProgLoss switch
    utils/tal.py               # STAL switch
    cfg/default.yaml           # new hyperparameters
    cfg/models/26/             # LA-YOLO26 model YAML files
  tools/
    lkyw_run.py                # grouped experiment launcher
    lkyw_prepare_splits.py     # dataset validation and no-AI split generation
    lkyw_collect_results.py    # result CSV aggregation
    lkyw_validate.py           # external validation and fire/no-fire metrics
    lkyw_dataset_stats.py      # dataset statistics for paper placeholders
  lkyw_experiments/
    plans/paper_full.yaml      # complete paper experiment plan
    datasets/*.yaml            # LKYWDetection, no-AI, Ext-VFSmoke, CarCrash YAMLs
    manifests/ai_images.txt    # AI-assisted training image list
    splits/*.txt               # generated train/val/test/no-AI splits
  datasets/
    LKYWDetection/
    CarCrashDetection/
    Ext-VFSmoke/               # currently a placeholder until images/labels are added
  weights/                     # YOLO11, YOLO26, YOLOv10 pretrained weights
  runs/                        # generated after training
```

## Current Dataset Status

`LKYWDetection` is ready:

```text
train: 968 images, 968 labels, 988 boxes
val:   277 images, 277 labels, 283 boxes
test:  141 images, 141 labels, 142 boxes
total: 1386 images, 1413 boxes
```

The AI-assisted training manifest resolves 147 training images, so `lkyw_experiments/splits/train_noai.txt` contains 821 non-AI training images.

`CarCrashDetection` is ready:

```text
train: 5321 images, 5321 labels
valid: 998 images, 998 labels
test:  333 images, 333 labels
```

`Ext-VFSmoke` exists only as a directory placeholder. Add YOLO-format images and labels under:

```text
datasets/Ext-VFSmoke/images/test
datasets/Ext-VFSmoke/labels/test
datasets/Ext-VFSmoke/images/smoke
datasets/Ext-VFSmoke/labels/smoke
```

## Implementation Changes

### 1. SFGA and C3k2SFGA

Files:

```text
ultralytics/nn/modules/block.py:1111
```

Key code:

```python
class SFGA(nn.Module):
    """Smoke-Flame Guided Attention for smoke/fire occlusion and tiny flame cues."""

    def __init__(self, c1: int, c2: int, k: int = 7, reduction: int = 16, residual: bool = True):
        self.proj = Conv(c1, c2, 1, 1) if c1 != c2 else nn.Identity()
        hidden = max(c2 // reduction, 8)
        self.channel_mlp = nn.Sequential(
            nn.Conv2d(c2, hidden, 1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, c2, 1, bias=True),
        )
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=k, stride=1, padding=k // 2, bias=False),
            nn.Sigmoid(),
        )
        self.edge_attn = nn.Sequential(
            nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1, groups=c2, bias=False),
            nn.Conv2d(c2, c2, kernel_size=1, stride=1, bias=True),
            nn.Sigmoid(),
        )
        self.gamma = nn.Parameter(torch.zeros(1))
```

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = self.proj(x)
    channel_gate = torch.sigmoid(self.channel_mlp(avg_pool) + self.channel_mlp(max_pool))
    spatial_gate = self.spatial_attn(torch.cat((avg_map, max_map), dim=1))
    edge_gate = self.edge_attn((x - local_mean).abs())
    y = x * channel_gate * spatial_gate * edge_gate
    return x + self.gamma * y if self.residual else y
```

```python
class C3k2SFGA(C3k2):
    """C3k2 block followed by SFGA for LA-YOLO26 neck features."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, attn=False, g=1, shortcut=True, k=7, reduction=16):
        super().__init__(c1, c2, n, c3k, e, attn, g, shortcut)
        self.sfga = SFGA(c2, c2, k=k, reduction=reduction, residual=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sfga(super().forward(x))
```

Explanation:

SFGA is a lightweight neck attention module designed for vehicle fire and smoke scenes. It combines channel attention, spatial attention, and local edge-guided attention. The local edge branch uses the absolute difference between the feature map and a local average map, which emphasizes small flames, smoke boundaries, and high-contrast fire-like regions. `C3k2SFGA` keeps YOLO26's original `C3k2` structure and appends SFGA, making it easy to replace neck blocks without changing the detection head.

### 2. YAML Parsing Support for the New Modules

Files:

```text
ultralytics/nn/modules/__init__.py
ultralytics/nn/tasks.py
```

Export changes:

```python
from .block import (
    C3k2,
    C3k2SFGA,
    SFGA,
)

__all__ = (
    "C3k2",
    "C3k2SFGA",
    "SFGA",
)
```

Parser changes:

```python
base_modules = frozenset(
    {
        C3k2,
        C3k2SFGA,
        SFGA,
    }
)

repeat_modules = frozenset(
    {
        C3k2,
        C3k2SFGA,
    }
)
```

```python
if m in frozenset({C3k2, C3k2SFGA}):  # for M/L/X sizes
    legacy = False
    if scale in "mlx":
        args[3] = True
```

Explanation:

Ultralytics builds models by reading the model YAML and resolving module names through the task parser. Adding `SFGA` and `C3k2SFGA` to the export list and parser sets allows model YAML files such as `la-yolo26m.yaml` to instantiate these modules normally. Adding `C3k2SFGA` to `repeat_modules` preserves the standard depth scaling behavior, and treating it with `C3k2` keeps YOLO26's scale-specific `c3k` behavior for M/L/X variants.

### 3. Focal Modulation and Fire/No-Fire Priority Loss

Files:

```text
ultralytics/utils/loss.py:369
ultralytics/cfg/default.yaml:107
```

Hyperparameter registration:

```python
self.fl_gamma = float(getattr(h, "fl_gamma", 0.0) or 0.0)
self.fl_alpha = float(getattr(h, "fl_alpha", 0.25) or 0.25)
self.fire_prior = float(getattr(h, "fire_prior", 0.0) or 0.0)
self.fire_class_ids = [i for i in self._parse_class_ids(getattr(h, "fire_class_ids", "")) if 0 <= i < self.nc]
self.nofire_class_ids = [i for i in self._parse_class_ids(getattr(h, "nofire_class_ids", "")) if 0 <= i < self.nc]
```

Focal modulation:

```python
cls_target = target_scores.to(dtype)
bce_loss = self.bce(pred_scores, cls_target)
if self.fl_gamma > 0:
    pred_prob = pred_scores.sigmoid()
    p_t = cls_target * pred_prob + (1.0 - cls_target) * (1.0 - pred_prob)
    focal_weight = (1.0 - p_t).clamp(min=0).pow(self.fl_gamma)
    if self.fl_alpha > 0:
        alpha_weight = cls_target * self.fl_alpha + (1.0 - cls_target) * (1.0 - self.fl_alpha)
        focal_weight = focal_weight * alpha_weight
    bce_loss = bce_loss * focal_weight
```

Fire/no-fire state priority loss:

```python
if self.fire_prior > 0 and fg_mask.sum() and self.fire_class_ids and self.nofire_class_ids:
    pos_logits = pred_scores[fg_mask]
    pos_labels = target_labels[fg_mask].long()

    fire_logits = torch.logsumexp(pos_logits.index_select(1, fire_ids), dim=1)
    nofire_logits = torch.logsumexp(pos_logits.index_select(1, nofire_ids), dim=1)
    state_logits = torch.stack((nofire_logits, fire_logits), dim=1)

    fire_target = (pos_labels.unsqueeze(1) == fire_ids.view(1, -1)).any(dim=1).long()
    loss[1] = loss[1] + self.fire_prior * F.cross_entropy(state_logits, fire_target, reduction="mean")
```

Default config:

```yaml
fl_gamma: 0.0
fl_alpha: 0.25
fire_prior: 0.0
fire_class_ids: ""
nofire_class_ids: ""
```

Explanation:

Focal modulation is disabled by default because `fl_gamma=0.0`. When enabled, it down-weights easy examples and focuses the classification loss on hard or ambiguous samples. The fire/no-fire priority loss converts the four detection classes into two emergency states by aggregating logits with `logsumexp`. For LKYWDetection, fire classes are `0,2` and no-fire classes are `1,3`. This adds explicit semantic pressure against confusing fire and no-fire vehicle states.

### 4. STAL and ProgLoss Ablation Switches

Files:

```text
ultralytics/cfg/default.yaml:33
ultralytics/utils/loss.py:389
ultralytics/utils/tal.py:40
ultralytics/utils/loss.py:1225
```

Default switches:

```yaml
stal: True
prog_loss: True
```

STAL is passed into the task-aligned assigner:

```python
self.assigner = TaskAlignedAssigner(
    topk=tal_topk,
    num_classes=self.nc,
    alpha=0.5,
    beta=6.0,
    stride=self.stride.tolist(),
    topk2=tal_topk2,
    small_target=bool(getattr(h, "stal", True)),
)
```

STAL controls tiny-GT expansion in `TaskAlignedAssigner`:

```python
if self.small_target:
    gt_bboxes_xywh = xyxy2xywh(gt_bboxes)
    wh_mask = gt_bboxes_xywh[..., 2:] < self.stride[0]
    gt_bboxes_xywh[..., 2:] = torch.where(
        (wh_mask * mask_gt).bool(),
        torch.tensor(self.stride_val, dtype=gt_bboxes_xywh.dtype, device=gt_bboxes_xywh.device),
        gt_bboxes_xywh[..., 2:],
    )
    gt_bboxes = xywh2xyxy(gt_bboxes_xywh)
```

ProgLoss controls the one-to-many / one-to-one decay schedule:

```python
self.prog_loss = bool(getattr(model.args, "prog_loss", True))

def update(self) -> None:
    if not self.prog_loss:
        return
    self.updates += 1
    self.o2m = self.decay(self.updates)
    self.o2o = max(self.total - self.o2m, 0)
```

Explanation:

The defaults keep YOLO26's official behavior: `stal=True` and `prog_loss=True`. The switches exist for ablation only. Setting `stal=False` disables small-target-aware label assignment expansion. Setting `prog_loss=False` freezes the one-to-many / one-to-one loss weights instead of applying the progressive schedule.

### 5. LA-YOLO26 Model YAML

Files:

```text
ultralytics/cfg/models/26/la-yolo26.yaml
ultralytics/cfg/models/26/la-yolo26n.yaml
ultralytics/cfg/models/26/la-yolo26s.yaml
ultralytics/cfg/models/26/la-yolo26m.yaml
```

Key YAML:

```yaml
nc: 4
end2end: True
reg_max: 1

head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 6], 1, Concat, [1]]
  - [-1, 2, C3k2, [512, True]]

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 4], 1, Concat, [1]]
  - [-1, 2, C3k2SFGA, [256, True]] # P3/8 small-object smoke-flame cues

  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 13], 1, Concat, [1]]
  - [-1, 2, C3k2SFGA, [512, True]] # P4/16 vehicle-smoke coupling

  - [[16, 19, 22], 1, Detect, [nc]]
```

Explanation:

LA-YOLO26 keeps YOLO26's end-to-end detection head and inserts `C3k2SFGA` into the P3 and P4 neck branches. P3/8 is used for tiny flames and small smoke cues. P4/16 is used for medium-scale vehicle-smoke coupling. The copied `la-yolo26n/s/m.yaml` aliases make the experiment plan explicit and allow Ultralytics to infer the intended scale from the file name.

### 6. Experiment Plan, Dataset YAMLs, and Four-PC Grouping

Files:

```text
lkyw_experiments/plans/paper_full.yaml
lkyw_experiments/datasets/lkyw.yaml
lkyw_experiments/datasets/lkyw_noai.yaml
lkyw_experiments/datasets/ext_vfsmoke.yaml
lkyw_experiments/datasets/ext_vfsmoke_smoke.yaml
lkyw_experiments/datasets/carcrash.yaml
tools/lkyw_run.py
```

Plan root:

```yaml
workspace: "{repo}"
weights_dir: "{workspace}/weights"
project: "{workspace}/runs/lkyw"

data:
  lkyw: "{repo}/lkyw_experiments/datasets/lkyw.yaml"
  lkyw_noai: "{repo}/lkyw_experiments/datasets/lkyw_noai.yaml"
  ext_vfsmoke: "{repo}/lkyw_experiments/datasets/ext_vfsmoke.yaml"
  ext_vfsmoke_smoke: "{repo}/lkyw_experiments/datasets/ext_vfsmoke_smoke.yaml"
  carcrash: "{repo}/lkyw_experiments/datasets/carcrash.yaml"

seeds: [0, 1, 42, 3407, 205]
```

Full loss preset:

```yaml
full_loss:
  fl_gamma: 1.5
  fl_alpha: 0.25
  fire_prior: 0.20
  fire_class_ids: "0,2"
  nofire_class_ids: "1,3"
```

Dataset YAML example:

```yaml
train: ../../datasets/LKYWDetection/images/train
val: ../../datasets/LKYWDetection/images/val
test: ../../datasets/LKYWDetection/images/test

names:
  0: carFire
  1: carNofire
  2: lkywFire
  3: lkywNofire
```

`lkyw_run.py` resolves the portable root and supports group selection:

```python
REPO = Path(__file__).resolve().parents[1]
os.environ.setdefault("YOLO_CONFIG_DIR", str(REPO / ".ultralytics_config"))

parser.add_argument("--group", nargs="*", help="Run experiments that contain any of these groups.")
parser.add_argument("--workspace", default=os.environ.get("LKYW_ROOT"), help="Override workspace root.")
parser.add_argument("--dry-run", action="store_true", help="Print resolved jobs without training.")

workspace_template = args.workspace or plan.get("workspace", repo_root)
workspace = render(workspace_template, {"repo": repo_root})
context = {"repo": repo_root, "workspace": Path(workspace).as_posix()}
```

Explanation:

`paper_full.yaml` is the single experiment source of truth. It defines shared defaults, augmentation presets, loss presets, model/data paths, seeds, and experiment groups. `tools/lkyw_run.py` reads the plan, resolves all paths from the copied package root, and runs only the experiments whose `groups` match the selected computer.

Four-machine split:

The default plan uses `seeds: [0, 1, 42, 3407, 205]`. The task count below means experiment definitions; actual run count is task count x 5. Use `--seed 0 --dry-run` for a quick check.

| Group | Tasks | Runs | Main purpose |
|---|---:|---:|---|
| `pc1` | 4 | 20 | Table 4 N/S-scale YOLO11 vs LA-YOLO26 comparison |
| `pc2` | 2 | 10 | Table 4 M-scale main comparison and LA-YOLO26M checkpoint source |
| `pc3` | 6 | 30 | Table 5 staged YOLO26M training-mechanism ablations |
| `pc4` | 8 | 40 | Table 5 LA-YOLO26M/no-AI/module ablations and Table 7 CarCrash transfer |

`pc1` runs Table 4 N/S-scale comparisons:

| Experiment | Model/weights | Data | Key settings | Paper role |
|---|---|---|---|---|
| `yolo11n_lkyw` | `weights/yolo11n.pt` | `lkyw` | `SGD`, physical aug, `stal=False`, `prog_loss=False` | YOLO11-N baseline |
| `yolo11s_lkyw` | `weights/yolo11s.pt` | `lkyw` | `SGD`, physical aug, `stal=False`, `prog_loss=False` | YOLO11-S baseline |
| `la_yolo26n_full` | `la-yolo26n.yaml` + `yolo26n.pt` | `lkyw` | `MuSGD`, SFGA, physical aug, STAL, ProgLoss, Focal, FirePrior | Full LA-YOLO26-N |
| `la_yolo26s_full` | `la-yolo26s.yaml` + `yolo26s.pt` | `lkyw` | `MuSGD`, SFGA, physical aug, STAL, ProgLoss, Focal, FirePrior | Full LA-YOLO26-S |

`pc2` runs Table 4 M-scale comparisons:

| Experiment | Model/weights | Data | Key settings | Paper role |
|---|---|---|---|---|
| `yolo11m_lkyw` | `weights/yolo11m.pt` | `lkyw` | `SGD`, physical aug, `stal=False`, `prog_loss=False` | YOLO11-M baseline |
| `la_yolo26m_full` | `la-yolo26m.yaml` + `yolo26m.pt` | `lkyw` | `MuSGD`, SFGA, physical aug, STAL, ProgLoss, Focal, FirePrior | Full LA-YOLO26-M |

`pc3` runs Table 5 YOLO26M staged ablations:

| Experiment | Added/changed item | Key settings | Paper role |
|---|---|---|---|
| `yolo26m_base` | Base YOLO26M | no physical aug, `SGD`, `stal=False`, `prog_loss=False` | original training baseline |
| `yolo26m_physical_aug` | physical augmentation | `scale/perspective/shear/mosaic/mixup` | physical augmentation gain |
| `yolo26m_physical_stal` | STAL | `stal=True`, `prog_loss=False` | small-target assignment gain |
| `yolo26m_physical_stal_focal` | Focal | `fl_gamma=1.5`, `fl_alpha=0.25` | hard-sample classification gain |
| `yolo26m_physical_stal_focal_prog` | ProgLoss | `prog_loss=True` | progressive one-to-many/one-to-one schedule gain |
| `yolo26m_physical_stal_focal_prog_musgd` | MuSGD | `optimizer=MuSGD` | final optimizer gain |

`pc4` runs LA-YOLO26M extended ablations and CarCrashDetection transfer:

| Experiment | Model/weights | Data | Key settings | Paper role |
|---|---|---|---|---|
| `la_yolo26m_sfga_noai` | `la-yolo26m.yaml` + `yolo26m.pt` | `lkyw_noai` | no AI-assisted training images, SFGA, STAL, ProgLoss, Focal | Table 5 no-AI ablation |
| `la_yolo26m_full_ablation` | `la-yolo26m.yaml` + `yolo26m.pt` | `lkyw` | SFGA, physical aug, STAL, ProgLoss, Focal, FirePrior, MuSGD | Table 5 full LA-YOLO26M item |
| `la_yolo26m_sfga_only` | `la-yolo26m.yaml` + `yolo26m.pt` | `lkyw` | SFGA, physical aug, STAL, ProgLoss, MuSGD, no Focal/FirePrior | SFGA baseline supplement |
| `la_yolo26m_sfga_focal` | `la-yolo26m.yaml` + `yolo26m.pt` | `lkyw` | SFGA plus Focal | Focal single-factor supplement |
| `la_yolo26m_sfga_fireprior` | `la-yolo26m.yaml` + `yolo26m.pt` | `lkyw` | SFGA plus FirePrior | fire/no-fire priority single-factor supplement |
| `la_yolo26n_carcrash` | `la-yolo26n.yaml` + `yolo26n.pt` | `carcrash` | LA-YOLO26-N on CarCrashDetection | Table 7 N-scale transfer |
| `la_yolo26s_carcrash` | `la-yolo26s.yaml` + `yolo26s.pt` | `carcrash` | LA-YOLO26-S on CarCrashDetection | Table 7 S-scale transfer |
| `la_yolo26m_carcrash` | `la-yolo26m.yaml` + `yolo26m.pt` | `carcrash` | LA-YOLO26-M on CarCrashDetection | Table 7 M-scale transfer |

Ext-VFSmoke is not a four-machine training job. It is an external validation task after training; populate `datasets/Ext-VFSmoke` first, then run `tools/lkyw_validate.py` for Table 6.

### 7. Helper Scripts

Files:

```text
tools/lkyw_prepare_splits.py
tools/lkyw_run.py
tools/lkyw_collect_results.py
tools/lkyw_validate.py
tools/lkyw_dataset_stats.py
```

Script roles:

```text
lkyw_prepare_splits.py
  Validates LKYWDetection image/label pairing, checks class order, resolves ai_images.txt,
  and writes train.txt, val.txt, test.txt, train_noai.txt, and lkyw_detection_stats.json.

lkyw_run.py
  Loads paper_full.yaml, selects experiments by --group or --only, resolves portable paths,
  and calls YOLO(...).train(**train_args).

lkyw_collect_results.py
  Reads runs/lkyw/*/results.csv, selects the best mAP50-95 row per run,
  and writes per-run plus mean/std summaries.

lkyw_validate.py
  Runs model.val() and computes fire/no-fire state metrics for external datasets.

lkyw_dataset_stats.py
  Counts images, boxes, class distribution, fire boxes, and small fire targets
  for dataset description placeholders.
```

## Quick Check

Run from the copied package root:

```powershell
cd <copied LAYOLO26 path>
D:/CondaEnvs/yolov11_traffic_dev/python.exe tools/lkyw_prepare_splits.py --require-ai
D:/CondaEnvs/yolov11_traffic_dev/python.exe tools/lkyw_run.py --group pc1 --seed 0 --dry-run
```

## Four-Machine Run

Run one command per computer from the copied package root:

```powershell
D:/CondaEnvs/yolov11_traffic_dev/python.exe tools/lkyw_run.py --group pc1
D:/CondaEnvs/yolov11_traffic_dev/python.exe tools/lkyw_run.py --group pc2
D:/CondaEnvs/yolov11_traffic_dev/python.exe tools/lkyw_run.py --group pc3
D:/CondaEnvs/yolov11_traffic_dev/python.exe tools/lkyw_run.py --group pc4
```

Use `--dry-run` first to inspect resolved jobs without training:

```powershell
D:/CondaEnvs/yolov11_traffic_dev/python.exe tools/lkyw_run.py --group pc4 --seed 0 --dry-run
```

If the copied folder is not the working directory, either run from that folder or pass:

```powershell
D:/CondaEnvs/yolov11_traffic_dev/python.exe tools/lkyw_run.py --group pc1 --workspace D:/LAYOLO26
```

## Result Collection

After all computers finish, copy their `runs/lkyw/*` folders back into one package and run:

```powershell
D:/CondaEnvs/yolov11_traffic_dev/python.exe tools/lkyw_collect_results.py --runs runs/lkyw
```

Generated summaries:

```text
runs/lkyw/lkyw_per_run_summary.csv
runs/lkyw/lkyw_mean_std_summary.csv
```

## External Validation

After Ext-VFSmoke is populated, validate the best checkpoint:

```powershell
D:/CondaEnvs/yolov11_traffic_dev/python.exe tools/lkyw_validate.py `
  --model runs/lkyw/la_yolo26m_full_seed0/weights/best.pt `
  --data lkyw_experiments/datasets/ext_vfsmoke.yaml `
  --split test `
  --out runs/lkyw/ext_vfsmoke_la_yolo26m_seed0.json
```

For the smoke subset:

```powershell
D:/CondaEnvs/yolov11_traffic_dev/python.exe tools/lkyw_validate.py `
  --model runs/lkyw/la_yolo26m_full_seed0/weights/best.pt `
  --data lkyw_experiments/datasets/ext_vfsmoke_smoke.yaml `
  --split test `
  --out runs/lkyw/ext_vfsmoke_smoke_la_yolo26m_seed0.json
```
