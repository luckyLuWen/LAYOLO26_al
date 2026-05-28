# LA-YOLO26 可迁移实验包

本目录是基于 Ultralytics 改造后的 LA-YOLO26 训练实验包，用于支撑 LKYWDetection 论文实验。
以下D:/CondaEnvs/yolov11_traffic_dev/python.exe改为自己的conda 环境的python.exe

## 目录结构

```text
LAYOLO26/
  ultralytics/                 # 已修改的 Ultralytics 主代码
    nn/modules/block.py        # SFGA 与 C3k2SFGA 模块
    nn/modules/__init__.py     # 新模块导出
    nn/tasks.py                # 模型 YAML 解析链路支持新模块
    utils/loss.py              # Focal 调制、火灾/无火状态优先损失、ProgLoss 开关
    utils/tal.py               # STAL 开关
    cfg/default.yaml           # 新增训练超参数
    cfg/models/26/             # LA-YOLO26 模型配置
  tools/
    lkyw_run.py                # 分组训练入口
    lkyw_prepare_splits.py     # 数据集校验与 no-AI 划分生成
    lkyw_collect_results.py    # 训练结果汇总
    lkyw_validate.py           # 外部验证与火灾/无火状态指标计算
    lkyw_dataset_stats.py      # 论文占位符所需的数据统计
  lkyw_experiments/
    plans/paper_full.yaml      # 完整论文实验计划
    datasets/*.yaml            # LKYWDetection、no-AI、Ext-VFSmoke、CarCrash 数据配置
    manifests/ai_images.txt    # AI 辅助生成训练图像清单
    splits/*.txt               # 已生成的 train/val/test/no-AI 划分文件
  datasets/
    LKYWDetection/
    CarCrashDetection/
    Ext-VFSmoke/               # 当前仍为空目录，需要后续补充图像和标签
  weights/                     # YOLO11、YOLO26、YOLOv10 预训练权重
  runs/                        # 训练后生成
```

## 当前数据集状态

`LKYWDetection` 已可直接训练：

```text
train: 968 张图像，968 个标签文件，988 个目标框
val:   277 张图像，277 个标签文件，283 个目标框
test:  141 张图像，141 个标签文件，142 个目标框
total: 1386 张图像，1413 个目标框
```

AI 辅助训练图像清单已解析出 147 张训练图，因此 `lkyw_experiments/splits/train_noai.txt` 中包含 821 张非 AI 训练图，可用于 no-AI 消融实验。

`CarCrashDetection` 已可直接运行迁移实验：

```text
train: 5321 张图像，5321 个标签文件
valid: 998 张图像，998 个标签文件
test:  333 张图像，333 个标签文件
```

`Ext-VFSmoke` 目前只是占位目录，还没有实际图像和 YOLO 标签。后续应按下面结构补齐：

```text
datasets/Ext-VFSmoke/images/test
datasets/Ext-VFSmoke/labels/test
datasets/Ext-VFSmoke/images/smoke
datasets/Ext-VFSmoke/labels/smoke
```

## 代码改动说明

### 1. 新增 SFGA 与 C3k2SFGA

文件位置：

```text
ultralytics/nn/modules/block.py:1111
```

关键代码：

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

解释：

SFGA 是面向车辆火灾、烟雾遮挡、小火焰目标设计的轻量注意力模块。它由通道注意力、空间注意力和局部边缘引导注意力组成。局部边缘分支通过特征图与局部均值图之间的绝对差异突出烟雾边缘、小火焰轮廓和强对比火焰区域。`C3k2SFGA` 保留 YOLO26 原有 `C3k2` 结构，仅在其输出后接入 SFGA，因此对检测头和主干结构的侵入较小，适合做消融对比。

### 2. YOLO26 模型解析链路支持新模块

文件位置：

```text
ultralytics/nn/modules/__init__.py
ultralytics/nn/tasks.py
```

模块导出：

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

模型 YAML 解析注册：

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

解释：

Ultralytics 通过读取模型 YAML 中的模块名来动态构建网络。如果只在 `block.py` 中写了新模块，但没有加入 `__init__.py` 和 `tasks.py` 的解析链路，模型 YAML 中的 `C3k2SFGA` 无法被识别。这里将 `SFGA` 与 `C3k2SFGA` 加入导出列表、基础模块集合和重复模块集合，使其能够参与深度缩放、通道缩放和重复次数解析。`C3k2SFGA` 与 `C3k2` 共用 M/L/X 规模下的 `c3k` 行为，保证新模块与 YOLO26 原有缩放逻辑一致。

### 3. 检测损失加入 Focal 调制与火灾/无火状态优先损失

文件位置：

```text
ultralytics/utils/loss.py:369
ultralytics/cfg/default.yaml:107
```

新增超参数读取：

```python
self.fl_gamma = float(getattr(h, "fl_gamma", 0.0) or 0.0)
self.fl_alpha = float(getattr(h, "fl_alpha", 0.25) or 0.25)
self.fire_prior = float(getattr(h, "fire_prior", 0.0) or 0.0)
self.fire_class_ids = [i for i in self._parse_class_ids(getattr(h, "fire_class_ids", "")) if 0 <= i < self.nc]
self.nofire_class_ids = [i for i in self._parse_class_ids(getattr(h, "nofire_class_ids", "")) if 0 <= i < self.nc]
```

Focal 调制：

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

火灾/无火状态优先损失：

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

默认配置：

```yaml
fl_gamma: 0.0
fl_alpha: 0.25
fire_prior: 0.0
fire_class_ids: ""
nofire_class_ids: ""
```

解释：

`fl_gamma=0.0` 时 Focal 调制关闭，默认不改变 YOLO26 的分类损失行为。开启后，Focal 调制会降低易分类样本对损失的贡献，使训练更关注难样本和边界模糊样本。火灾/无火状态优先损失将四分类检测结果进一步映射为“无火”和“有火”两个语义状态，通过 `logsumexp` 聚合同状态类别的 logits。对于 LKYWDetection，火灾类为 `0,2`，无火类为 `1,3`。该损失用于降低车辆火灾识别中“有火/无火”状态混淆的风险。

### 4. 为 STAL 与 ProgLoss 增加消融开关

文件位置：

```text
ultralytics/cfg/default.yaml:33
ultralytics/utils/loss.py:389
ultralytics/utils/tal.py:40
ultralytics/utils/loss.py:1225
```

默认开关：

```yaml
stal: True
prog_loss: True
```

STAL 传入任务对齐分配器：

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

STAL 控制小目标 GT 扩张：

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

ProgLoss 控制一对多/一对一损失权重调度：

```python
self.prog_loss = bool(getattr(model.args, "prog_loss", True))

def update(self) -> None:
    if not self.prog_loss:
        return
    self.updates += 1
    self.o2m = self.decay(self.updates)
    self.o2o = max(self.total - self.o2m, 0)
```

解释：

默认值 `stal=True`、`prog_loss=True` 保持 YOLO26 官方行为不变。新增开关主要服务于论文消融实验：`stal=False` 用于关闭小目标感知标签分配扩张，`prog_loss=False` 用于关闭一对多/一对一损失的渐进式权重调度，使其权重保持固定。

### 5. 新增 LA-YOLO26 模型配置

文件位置：

```text
ultralytics/cfg/models/26/la-yolo26.yaml
ultralytics/cfg/models/26/la-yolo26n.yaml
ultralytics/cfg/models/26/la-yolo26s.yaml
ultralytics/cfg/models/26/la-yolo26m.yaml
```

关键 YAML：

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

解释：

LA-YOLO26 保留 YOLO26 的端到端检测头，并在颈部 P3 和 P4 分支引入 `C3k2SFGA`。P3/8 负责增强小火焰和细粒度烟雾线索，P4/16 负责增强车辆主体与烟雾、火焰之间的中尺度关联。`la-yolo26n/s/m.yaml` 是与共享结构一致的显式别名文件，便于实验计划中直接指定 N/S/M 三种规模，也便于 Ultralytics 根据文件名推断模型尺度。

### 6. 新增完整实验计划、数据集 YAML 与四台电脑分组脚本

文件位置：

```text
lkyw_experiments/plans/paper_full.yaml
lkyw_experiments/datasets/lkyw.yaml
lkyw_experiments/datasets/lkyw_noai.yaml
lkyw_experiments/datasets/ext_vfsmoke.yaml
lkyw_experiments/datasets/ext_vfsmoke_smoke.yaml
lkyw_experiments/datasets/carcrash.yaml
tools/lkyw_run.py
```

实验计划根配置：

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

完整损失预设：

```yaml
full_loss:
  fl_gamma: 1.5
  fl_alpha: 0.25
  fire_prior: 0.20
  fire_class_ids: "0,2"
  nofire_class_ids: "1,3"
```

数据集 YAML 示例：

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

`lkyw_run.py` 负责解析可迁移根目录并按组选择实验：

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

解释：

`paper_full.yaml` 是论文实验的统一配置入口，集中定义默认训练参数、数据增强预设、损失预设、模型路径、数据路径、随机种子和实验分组。`tools/lkyw_run.py` 读取该计划文件后，会以当前拷贝的 `LAYOLO26` 根目录作为 `{repo}`，自动解析权重、数据集和输出目录路径，并只运行所选 `--group` 对应的实验。

四台电脑分工：

默认每个实验都会跑 `seeds: [0, 1, 42, 3407, 205]` 五个随机种子。下面的“任务数”指实验配置数量；实际 run 数 = 任务数 x 5。如果只是检查流程，使用 `--seed 0 --dry-run`。

| 分组 | 任务数 | 实际 run 数 | 主要补充内容 |
|---|---:|---:|---|
| `pc1` | 4 | 20 | 补齐表 4 中 N/S 小模型尺度的 YOLO11 与 LA-YOLO26 对比 |
| `pc2` | 2 | 10 | 补齐表 4 中 M 尺度主模型对比，并产出后续外部验证常用的 LA-YOLO26M checkpoint |
| `pc3` | 6 | 30 | 补齐表 5 中 YOLO26M 的阶段式训练机制消融 |
| `pc4` | 8 | 40 | 补齐表 5 的 LA-YOLO26M/no-AI/模块消融，以及表 7 的 CarCrashDetection 迁移实验 |

`pc1` 运行表 4 的 N/S 尺度对比实验：

| 实验名 | 模型/权重 | 数据集 | 关键设置 | 论文作用 |
|---|---|---|---|---|
| `yolo11n_lkyw` | `weights/yolo11n.pt` | `lkyw` | `SGD`，物理增强，`stal=False`，`prog_loss=False` | YOLO11-N 基线 |
| `yolo11s_lkyw` | `weights/yolo11s.pt` | `lkyw` | `SGD`，物理增强，`stal=False`，`prog_loss=False` | YOLO11-S 基线 |
| `la_yolo26n_full` | `la-yolo26n.yaml` + `yolo26n.pt` | `lkyw` | `MuSGD`，SFGA，物理增强，STAL，ProgLoss，Focal，FirePrior | LA-YOLO26-N 完整模型 |
| `la_yolo26s_full` | `la-yolo26s.yaml` + `yolo26s.pt` | `lkyw` | `MuSGD`，SFGA，物理增强，STAL，ProgLoss，Focal，FirePrior | LA-YOLO26-S 完整模型 |

`pc1` 补充的是轻量级和小模型尺度实验，用于说明 LA-YOLO26 在较低参数量/计算量下相对 YOLO11 的效果提升。

`pc2` 运行表 4 的 M 尺度主对比实验：

| 实验名 | 模型/权重 | 数据集 | 关键设置 | 论文作用 |
|---|---|---|---|---|
| `yolo11m_lkyw` | `weights/yolo11m.pt` | `lkyw` | `SGD`，物理增强，`stal=False`，`prog_loss=False` | YOLO11-M 基线 |
| `la_yolo26m_full` | `la-yolo26m.yaml` + `yolo26m.pt` | `lkyw` | `MuSGD`，SFGA，物理增强，STAL，ProgLoss，Focal，FirePrior | LA-YOLO26-M 完整模型 |

`pc2` 补充的是 M 尺度主模型对比。`la_yolo26m_full` 通常也是后续 Ext-VFSmoke 外部验证、可视化和论文主结论最重要的 checkpoint 来源。

`pc3` 运行表 5 的 YOLO26M 阶段式消融实验：

| 实验名 | 在上一项基础上新增/改变 | 关键设置 | 论文作用 |
|---|---|---|---|
| `yolo26m_base` | 基础 YOLO26M | 无物理增强，`SGD`，`stal=False`，`prog_loss=False` | 原始训练基线 |
| `yolo26m_physical_aug` | 加入物理增强 | `scale/perspective/shear/mosaic/mixup` | 验证事故场景物理增强收益 |
| `yolo26m_physical_stal` | 加入 STAL | `stal=True`，`prog_loss=False` | 验证小目标感知标签分配收益 |
| `yolo26m_physical_stal_focal` | 加入 Focal | `fl_gamma=1.5`，`fl_alpha=0.25` | 验证难样本分类调制收益 |
| `yolo26m_physical_stal_focal_prog` | 加入 ProgLoss | `prog_loss=True` | 验证渐进式一对多/一对一损失调度收益 |
| `yolo26m_physical_stal_focal_prog_musgd` | 优化器改为 MuSGD | `optimizer=MuSGD` | 验证最终优化器设置收益 |

`pc3` 不使用 LA-YOLO26 的 SFGA 结构，重点是把 YOLO26M 上的训练策略逐步拆开，形成“物理增强 -> STAL -> Focal -> ProgLoss -> MuSGD”的阶段式消融链。

`pc4` 运行 LA-YOLO26M 扩展消融和 CarCrashDetection 迁移实验：

| 实验名 | 模型/权重 | 数据集 | 关键设置 | 论文作用 |
|---|---|---|---|---|
| `la_yolo26m_sfga_noai` | `la-yolo26m.yaml` + `yolo26m.pt` | `lkyw_noai` | 去除 AI 辅助训练图，SFGA，STAL，ProgLoss，Focal | 表 5 no-AI 消融，验证 AI 辅助数据贡献 |
| `la_yolo26m_full_ablation` | `la-yolo26m.yaml` + `yolo26m.pt` | `lkyw` | SFGA，物理增强，STAL，ProgLoss，Focal，FirePrior，MuSGD | 表 5 LA-YOLO26M 完整项 |
| `la_yolo26m_sfga_only` | `la-yolo26m.yaml` + `yolo26m.pt` | `lkyw` | SFGA，物理增强，STAL，ProgLoss，MuSGD，不启用 Focal/FirePrior | 补充 SFGA 基础对照 |
| `la_yolo26m_sfga_focal` | `la-yolo26m.yaml` + `yolo26m.pt` | `lkyw` | 在 SFGA 对照上加入 Focal | 补充 Focal 单因素对照 |
| `la_yolo26m_sfga_fireprior` | `la-yolo26m.yaml` + `yolo26m.pt` | `lkyw` | 在 SFGA 对照上加入 FirePrior | 补充火灾/无火状态优先损失单因素对照 |
| `la_yolo26n_carcrash` | `la-yolo26n.yaml` + `yolo26n.pt` | `carcrash` | LA-YOLO26-N，CarCrashDetection 两类迁移 | 表 7 N 尺度迁移稳定性 |
| `la_yolo26s_carcrash` | `la-yolo26s.yaml` + `yolo26s.pt` | `carcrash` | LA-YOLO26-S，CarCrashDetection 两类迁移 | 表 7 S 尺度迁移稳定性 |
| `la_yolo26m_carcrash` | `la-yolo26m.yaml` + `yolo26m.pt` | `carcrash` | LA-YOLO26-M，CarCrashDetection 两类迁移 | 表 7 M 尺度迁移稳定性 |

`pc4` 的前三个模块消融用于补足论文中 SFGA、Focal、FirePrior 的单因素描述；后三个 CarCrash 实验用于证明模型迁移到外部交通事故严重程度数据集时的稳定性。Ext-VFSmoke 不在四台电脑的训练任务中，它是训练完成后的外部验证任务；需要先补齐 `datasets/Ext-VFSmoke` 图像和标签，再用 `tools/lkyw_validate.py` 运行表 6。

### 7. 新增训练、汇总、外部验证与数据统计脚本

文件位置：

```text
tools/lkyw_prepare_splits.py
tools/lkyw_run.py
tools/lkyw_collect_results.py
tools/lkyw_validate.py
tools/lkyw_dataset_stats.py
```

脚本职责：

```text
lkyw_prepare_splits.py
  校验 LKYWDetection 的图像/标签配对关系，检查类别顺序，解析 ai_images.txt，
  并生成 train.txt、val.txt、test.txt、train_noai.txt 和 lkyw_detection_stats.json。

lkyw_run.py
  读取 paper_full.yaml，按 --group 或 --only 选择实验，解析可迁移路径，
  并调用 YOLO(...).train(**train_args) 启动训练。

lkyw_collect_results.py
  读取 runs/lkyw/*/results.csv，按 mAP50-95 选择每个 run 的最佳行，
  生成单次实验结果表和均值/标准差汇总表。

lkyw_validate.py
  调用 model.val() 进行外部验证，并额外计算火灾/无火状态级指标。

lkyw_dataset_stats.py
  统计数据集图像数、目标框数、类别分布、火灾目标数和小火灾目标数，
  用于填充论文中的数据集描述占位符。
```

## 快速检查

在拷贝后的 `LAYOLO26` 根目录运行：

```powershell
cd <copied LAYOLO26 path>
D:/CondaEnvs/yolov11_traffic_dev/python.exe tools/lkyw_prepare_splits.py --require-ai
D:/CondaEnvs/yolov11_traffic_dev/python.exe tools/lkyw_run.py --group pc1 --seed 0 --dry-run
```

## 四台电脑并行训练

每台电脑进入自己的 `LAYOLO26` 根目录后，只运行对应组命令：

```powershell
D:/CondaEnvs/yolov11_traffic_dev/python.exe tools/lkyw_run.py --group pc1
D:/CondaEnvs/yolov11_traffic_dev/python.exe tools/lkyw_run.py --group pc2
D:/CondaEnvs/yolov11_traffic_dev/python.exe tools/lkyw_run.py --group pc3
D:/CondaEnvs/yolov11_traffic_dev/python.exe tools/lkyw_run.py --group pc4
```

pc1：4 个任务，20 个 run，补表 4 的 YOLO11 N/S 与 LA-YOLO26 N/S 对比。
pc2：2 个任务，10 个 run，补表 4 的 M 尺度主对比，并产出后续外部验证常用的 LA-YOLO26M checkpoint。
pc3：6 个任务，30 个 run，补表 5 的 YOLO26M 阶段式消融：物理增强、STAL、Focal、ProgLoss、MuSGD。
pc4：8 个任务，40 个 run，补表 5 的 no-AI/LA-YOLO26M/SFGA-Focal-FirePrior 消融，以及表 7 的 CarCrashDetection 迁移实验。

Ext-VFSmoke是训练完成后的表 6 外部验证任务，需要先补齐数据集后用 lkyw_validate.py 跑。

正式训练前建议先 dry-run：

```powershell
D:/CondaEnvs/yolov11_traffic_dev/python.exe tools/lkyw_run.py --group pc4 --seed 0 --dry-run
```

如果没有在 `LAYOLO26` 根目录运行，也可以显式指定工作目录：

```powershell
D:/CondaEnvs/yolov11_traffic_dev/python.exe tools/lkyw_run.py --group pc1 --workspace D:/LAYOLO26
```

## 结果汇总

所有电脑训练完成后，将各机器的 `runs/lkyw/*` 结果目录复制回同一个包中，然后运行：

```powershell
D:/CondaEnvs/yolov11_traffic_dev/python.exe tools/lkyw_collect_results.py --runs runs/lkyw
```

输出文件：

```text
runs/lkyw/lkyw_per_run_summary.csv
runs/lkyw/lkyw_mean_std_summary.csv
```

## 外部验证

补齐 Ext-VFSmoke 后，用最优 checkpoint 进行外部验证：

```powershell
D:/CondaEnvs/yolov11_traffic_dev/python.exe tools/lkyw_validate.py `
  --model runs/lkyw/la_yolo26m_full_seed0/weights/best.pt `
  --data lkyw_experiments/datasets/ext_vfsmoke.yaml `
  --split test `
  --out runs/lkyw/ext_vfsmoke_la_yolo26m_seed0.json
```

烟雾遮挡子集验证：

```powershell
D:/CondaEnvs/yolov11_traffic_dev/python.exe tools/lkyw_validate.py `
  --model runs/lkyw/la_yolo26m_full_seed0/weights/best.pt `
  --data lkyw_experiments/datasets/ext_vfsmoke_smoke.yaml `
  --split test `
  --out runs/lkyw/ext_vfsmoke_smoke_la_yolo26m_seed0.json
```
