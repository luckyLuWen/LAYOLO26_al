# LKYWDetection / LA-YOLO26 experiment pack

This folder contains the configs needed to fill the paper placeholders.

## Expected local layout

```text
<repo>/weights/yolo11n.pt
<repo>/weights/yolo11s.pt
<repo>/weights/yolo11m.pt
<repo>/weights/yolo26n.pt
<repo>/weights/yolo26s.pt
<repo>/weights/yolo26m.pt
<repo>/datasets/LKYWDetection
<repo>/datasets/Ext-VFSmoke
<repo>/datasets/CarCrashDetection
<repo>/runs/lkyw
```

Run commands from the package root. If a computer needs an explicit workspace root, set `LKYW_ROOT` before running:

```powershell
$env:LKYW_ROOT="E:/LAYOLO26"
```

`LKYWDetection` uses this class order from its `classes.txt` files:

```text
0 carFire
1 carNofire
2 lkywFire
3 lkywNofire
```

So fire-state loss uses `fire_class_ids="0,2"` and `nofire_class_ids="1,3"`.

Before training, `lkyw_experiments/manifests/ai_images.txt` must list the AI-assisted training images. The current
manifest contains 147 matched training images. Re-run this whenever the AI list changes:

```powershell
D:/CondaEnvs/yolov11_traffic_dev/python.exe tools/lkyw_prepare_splits.py --require-ai
```

This writes `<repo>/lkyw_experiments/splits/train_noai.txt` for the no-AI ablation and validates image/label pairing.

The external smoke set must be a separate test-only dataset under
`<repo>/datasets/Ext-VFSmoke`; do not mix it into `LKYWDetection/train`.

`Ext-VFSmoke` should be an external vehicle-centered smoke/fire robustness test set. Include images such as:

- vehicle accident or roadside vehicle fire with visible flame/smoke
- vehicle near dense smoke, dust, backlight, glare, or black body shadow that can confuse fire/no-fire judgment
- no-fire hard negatives around vehicles, for example smoke-like dust, emergency lights, reflections, headlights, or dark shadows

Avoid plain non-traffic smoke/fire images as the main content, such as only a forest fire, kitchen fire, building fire,
or a smoke plume with no vehicle target. A small number can appear only if a vehicle target is still the labeled object.
Use the same four classes as LKYWDetection:

```text
0 carFire
1 carNofire
2 lkywFire
3 lkywNofire
```

Recommended layout:

```text
datasets/Ext-VFSmoke/images/test
datasets/Ext-VFSmoke/labels/test
datasets/Ext-VFSmoke/images/smoke
datasets/Ext-VFSmoke/labels/smoke
```

Model paths such as `ultralytics/cfg/models/26/la-yolo26m.yaml` are intentional. Ultralytics extracts the `m` scale from
that name and loads the shared `la-yolo26.yaml` definition.

## Four-computer split

Run from the repository root:

```powershell
cd <copied LAYOLO26 path>

# PC 1: YOLO11 N/S and LA-YOLO26 N/S for table 4
D:/CondaEnvs/yolov11_traffic_dev/python.exe tools/lkyw_run.py --group pc1

# PC 2: YOLO11 M and LA-YOLO26 M for table 4
D:/CondaEnvs/yolov11_traffic_dev/python.exe tools/lkyw_run.py --group pc2

# PC 3: staged YOLO26M ablations for table 5
D:/CondaEnvs/yolov11_traffic_dev/python.exe tools/lkyw_run.py --group pc3

# PC 4: SFGA/Focal/FirePrior, no-AI, and CarCrash transfer runs
D:/CondaEnvs/yolov11_traffic_dev/python.exe tools/lkyw_run.py --group pc4
```

Use `--dry-run` first to print the resolved jobs without training. Use `--seed 0` to run only one seed while checking.

## Summaries

After copying all run folders back into `<repo>/runs/lkyw`, create mean/std tables:

```powershell
D:/CondaEnvs/yolov11_traffic_dev/python.exe tools/lkyw_collect_results.py --runs runs/lkyw
```

This writes:

```text
<repo>/runs/lkyw/lkyw_per_run_summary.csv
<repo>/runs/lkyw/lkyw_mean_std_summary.csv
```

## External validation

For table 6, run the best LA-YOLO26M checkpoint on Ext-VFSmoke:

```powershell
D:/CondaEnvs/yolov11_traffic_dev/python.exe tools/lkyw_validate.py `
  --model runs/lkyw/la_yolo26m_full_seed0/weights/best.pt `
  --data lkyw_experiments/datasets/ext_vfsmoke.yaml `
  --split test `
  --out runs/lkyw/ext_vfsmoke_la_yolo26m_seed0.json
```

For the smoke-occlusion subset recall, pass `lkyw_experiments/datasets/ext_vfsmoke_smoke.yaml`.

## Dataset counts

Use this for the Ext-VFSmoke dataset description placeholders:

```powershell
D:/CondaEnvs/yolov11_traffic_dev/python.exe tools/lkyw_dataset_stats.py `
  --data lkyw_experiments/datasets/ext_vfsmoke.yaml `
  --split test `
  --out runs/lkyw/ext_vfsmoke_stats.json
```

If you create optional subset folders named `images/smoke`, `images/small_fire`, and `images/night`, the script also
reports their image percentages.
