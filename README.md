# Corruption-Aware AV Perception

A camera-only autonomous-vehicle perception pipeline that detects and tracks road users,
segments drivable space, and estimates when its own perception is no longer trustworthy.

The project deliberately starts small. Milestone 1 only loads a real nuScenes camera frame,
projects its annotations, and explains the dataset records involved. CUDA, TensorRT, tracking,
segmentation, and corruption experiments come later.

## Milestone 1: inspect nuScenes mini

### 1. Create the environment

On an Apple Silicon Mac, run the bootstrap script. It installs `uv`, Python 3.11, and all
dependencies locally inside this project; it does not require Homebrew or modify shell profiles.

```bash
sh scripts/bootstrap_macos.sh
source .venv/bin/activate
python --version
pytest -q
```

### 2. Download nuScenes mini

Create a nuScenes account, download the **v1.0-mini** archive, and extract it locally. The
directory supplied to `--dataroot` must contain both `samples/` and `v1.0-mini/`:

```text
data/nuscenes/
├── maps/
├── samples/
├── sweeps/
└── v1.0-mini/
```

The dataset is ignored by Git. A symlink at `data/nuscenes` is also fine if the files live on
an external drive.

### 3. Inspect a front-camera sample

```bash
inspect-nuscenes \
  --dataroot data/nuscenes \
  --output outputs/nuscenes_cam_front.png
```

The command uses the first sample by default, prints the scene/sample/sample-data relationships,
and writes an annotated image. Useful options:

```bash
# Inspect a specific sample and show every projected annotation.
inspect-nuscenes --dataroot data/nuscenes \
  --sample-token <TOKEN> --all-annotations

# Inspect another camera and open an interactive window as well as saving the image.
inspect-nuscenes --dataroot data/nuscenes \
  --channel CAM_FRONT_LEFT --show
```

Run `inspect-nuscenes --help` for all options.

## Dataset vocabulary

- **Scene:** one roughly 20-second driving sequence.
- **Sample:** one annotated keyframe, normally spaced about 0.5 seconds apart.
- **Sample data:** a sensor observation associated with a sample, such as `CAM_FRONT`.
- **Sample annotation:** a labeled 3D object instance at one sample.
- **Instance:** the same physical object across samples in a scene.

nuScenes annotations are 3D boxes in the global coordinate system. The devkit transforms them
into the selected camera coordinate system and projects them using that camera's calibration.

## Repository layout

```text
src/av_perception/data/   Dataset loading and inspection
tests/                    Fast unit tests that do not require the dataset
outputs/                  Generated user-facing images and reports
```

## Milestone 2: object detection

Milestone 2 evaluates pretrained **YOLOv8s** COCO weights on nuScenes `CAM_FRONT`
keyframes. The detector is isolated behind an `ObjectDetector.predict_batch()` interface;
the dataset adapter, IoU matching, metrics, CSV writing, and visualization code do not import
Ultralytics. A later ONNX Runtime or TensorRT backend can therefore implement the same interface.

### Class mapping

| COCO class | COCO ID | Evaluation class |
|---|---:|---|
| person | 0 | pedestrian |
| car | 2 | vehicle |
| bus | 5 | vehicle |
| truck | 7 | vehicle |

Comparable nuScenes ground truth is mapped from `vehicle.car`, `vehicle.truck`,
`vehicle.bus.*`, and `human.pedestrian.*`. Bicycle, motorcycle, trailer, construction,
emergency, and other categories are excluded because they do not correspond exactly to the
requested COCO mapping.

### Ground-truth filtering

The default evaluator excludes:

- annotations with nuScenes visibility below `2` (less than approximately 40% visible);
- projected boxes smaller than `400 px²` after clipping to the camera image.

These thresholds are configurable with `--min-visibility` and `--min-box-area`. The defaults
reduce ambiguous, heavily occluded, and extremely small targets while preserving challenging
objects. Matching is greedy, class-aware, one-to-one, and uses IoU `>= 0.5` by default.

Exact default evaluation configuration:

| Setting | Value |
|---|---:|
| Detector confidence threshold | 0.25 |
| Match IoU threshold | 0.50 |
| Minimum nuScenes visibility token | 2 |
| Minimum clipped projected-box area | 400 px² |
| Inference image size | 640 px |

Ground-truth 3D boxes are transformed by the nuScenes devkit into `CAM_FRONT` camera
coordinates, projected with the calibrated camera intrinsic matrix, converted to the min/max
axis-aligned 2D rectangle, clipped to the image, then filtered. Predictions are sorted by
descending confidence and greedily assigned to the highest-IoU unmatched GT of the same mapped
class. A prediction is a true positive only when that IoU is at least 0.50; remaining predictions
are false positives and remaining GT boxes are false negatives.

AP uses an all-points interpolated precision-recall integral: predictions are confidence-ranked,
cumulative precision and recall are calculated, the precision envelope is made monotonically
non-increasing, and area is summed at each recall change. Reported mAP is the arithmetic mean of
vehicle AP and pedestrian AP. This is AP@0.5 for this project-specific filtered two-class task,
not nuScenes' official center-distance detection metric and not COCO AP@[.5:.95].

### Run the evaluator

```bash
source .venv/bin/activate

# Pilot on the first three scenes.
evaluate-detection \
  --dataroot data/nuscenes \
  --output-dir outputs/milestone2/pilot \
  --scenes 3 \
  --device mps

# Full nuScenes mini split (all ten scenes).
evaluate-detection \
  --dataroot data/nuscenes \
  --output-dir outputs/milestone2/full \
  --device mps
```

Each run writes:

- `detections.csv`: one long-form file containing detection and filtered-ground-truth rows;
- `metrics.json`: configuration, runtime, precision, recall, and AP per class;
- `visualizations/*.jpg`: green GT boxes, blue matched predictions, and red unmatched predictions.

The CSV retains scene/frame/sample identity, image path, normalized class, original source class,
confidence, coordinates, IoU, match status, annotation token, and visibility. This is the stable
schema that later corruption and reliability experiments will reuse.

Measured on all 404 nuScenes mini `CAM_FRONT` keyframes with batch size 8 (end-to-end evaluator
timing, including image loading, inference, projection/matching, and selected visualizations):

| Device | Total time | Latency/frame | Throughput |
|---|---:|---:|---:|
| CPU | 20.90 s | 51.73 ms | 19.33 FPS |
| Apple MPS | 10.09 s | 24.98 ms | 40.03 FPS |

MPS produced the same detection counts and matching results as CPU, with approximately 2.07×
higher end-to-end throughput.

### Ultralytics license

Ultralytics YOLO and its pretrained weights are distributed under the **AGPL-3.0 license** unless
a separate Ultralytics Enterprise license applies. This repository's use of the Ultralytics
package must comply with those terms. Review licensing before redistribution, network service
deployment, or commercial use. The backend boundary also allows replacement with a differently
licensed detector if required.

## Milestones 5–6: corruption engine and failure benchmark

The corruption benchmark applies five deterministic camera corruptions at severities 1–4.
Every stochastic transform derives its random generator from the configured base seed, sample
token, corruption name, and severity, so a specific corrupted frame is exactly reproducible.

| Corruption | Severity 1 → 4 parameterization |
|---|---|
| Fog | spatial haze alpha 0.16, 0.30, 0.45, 0.60 |
| Low light | exposure scale 0.62, 0.44, 0.29, 0.17 with increasing gamma |
| Blur | Gaussian kernels 5, 9, 15, 23 pixels |
| Noise | Gaussian sigma 8, 16, 28, 42 intensity levels |
| Partial occlusion | seeded near-black rectangle covering 8%, 16%, 28%, 40% |

Run a two-scene pilot or all ten mini scenes:

```bash
benchmark-corruptions \
  --dataroot data/nuscenes \
  --output-dir outputs/milestone56/pilot \
  --scenes 2 \
  --device mps \
  --seed 20261006

benchmark-corruptions \
  --dataroot data/nuscenes \
  --output-dir outputs/milestone56/full \
  --device mps \
  --seed 20261006
```

Each benchmark writes one `detections.csv` containing clean and corrupted rows with `corruption`,
`severity`, and per-frame `seed` columns, plus `metrics.json` and `ap_vs_severity.png`. Corrupted
frames are generated batch-by-batch in memory instead of permanently duplicating the dataset.

## Milestone 3: drivable-area segmentation

**nuScenes has no per-pixel drivable-area labels for camera images, so evaluate segmentation by
consistency, meaning IoU between each corrupted frame's mask and the clean frame's mask, and say
clearly in the README that it measures robustness and not accuracy.**

Accordingly, segmentation mask IoU in this project is a self-consistency/robustness measurement.
It does **not** establish that either the clean or corrupted mask is an accurate representation of
the road. Qualitative inspection of clean masks is reported separately, without presenting it as
ground-truth validation.

The implementation uses `nvidia/segformer-b0-finetuned-cityscapes-1024-1024` and extracts the
Cityscapes `road` class as a drivable-area proxy. The segmenter is wrapped behind
`Segmenter.segment_batch()`, which accepts paths or in-memory arrays and returns boolean masks at
the source resolution. This keeps the benchmark independent of PyTorch and permits future ONNX or
TensorRT backends.

For each frame, the clean predicted mask is the reference. The same frame is then processed under
all five deterministic corruptions at severities 1–4. The reported metric is binary mask IoU:
`intersection(clean, corrupted) / union(clean, corrupted)`. Results include per-frame IoU plus
per-condition mean, median, and 10th-percentile IoU. An empty clean mask and empty corrupted mask
receive IoU 1.0 because they agree, though clean-mask visualizations should be inspected to catch
consistently empty or implausible outputs.

```bash
# Two-scene pilot.
benchmark-segmentation \
  --dataroot data/nuscenes \
  --output-dir outputs/milestone3/pilot \
  --scenes 2 \
  --device mps

# Full ten-scene mini dataset.
benchmark-segmentation \
  --dataroot data/nuscenes \
  --output-dir outputs/milestone3/full \
  --device mps
```

Each run writes `segmentation_consistency.csv`, `metrics.json`, clean-mask qualitative overlays,
and `mask_iou_vs_severity.png`.

## Milestone 4: object tracking

Tracking uses the established Ultralytics implementation of **ByteTrack**, fed by the normalized
YOLOv8s vehicle/pedestrian detections. It is wrapped behind `MultiObjectTracker`, so the association
backend can be replaced later without changing evaluation or nuScenes loading. The tracker is reset
at each scene boundary and for every corruption/severity condition. The dependency and inherited
Ultralytics components are AGPL-3.0 licensed, as noted above.

The primary experiment deliberately uses annotated nuScenes CAM_FRONT keyframes. These are sampled
at approximately **2 Hz**, so an object can move substantially during the roughly 0.5 seconds
between evaluated frames. ByteTrack is normally used on much denser video and this sparse sampling
makes motion prediction and IoU association unusually difficult. The reported values are therefore
a constrained keyframe experiment and must not be presented as production-rate tracking results.
nuScenes camera sweeps could provide denser temporal input, but they do not have the same keyframe
annotation setup; they are excluded from the primary evaluation for now.

Ground-truth identities come directly from nuScenes `instance_token` values. The same visibility
(`>= 2`), projected-area (`>= 400 px²`), class mapping, detector confidence (`>= 0.25`), and
class-aware IoU matching threshold (`0.5`) used by detection evaluation apply here. ByteTrack uses
high/new-track thresholds `0.25`, low threshold `0.1`, match threshold `0.8`, score fusion, and a
six-keyframe track buffer. Tracking is restarted per scene.

Metrics are defined as follows:

- **IDF1:** a global Hungarian assignment maximizes matched frame observations between nuScenes
  instance tokens and scene-scoped predicted track IDs. `IDTP` is the assigned overlap count,
  `IDFN = GT observations - IDTP`, `IDFP = predicted observations - IDTP`, and
  `IDF1 = 2*IDTP / (2*IDTP + IDFP + IDFN)`.
- **ID switches:** a ground-truth identity is matched to a different predicted ID than at its
  previous matched observation.
- **Fragmentations:** a matched ground-truth identity becomes unmatched for one or more evaluated
  observations and is later matched again.
- **Observation retention:** matched filtered GT observations divided by all filtered GT
  observations. Mean identity retention averages that ratio per nuScenes instance.

Raw switch and fragmentation counts can decrease under severe corruption simply because the
detector returns too few tracks to switch. Interpret them together with IDF1 and retention, not as
standalone evidence that severe corruption improves tracking.

```bash
# Two-scene pilot, followed by all ten mini scenes after validation.
benchmark-tracking \
  --dataroot data/nuscenes \
  --output-dir outputs/milestone4/pilot \
  --scenes 2 \
  --device mps \
  --seed 20261006

benchmark-tracking \
  --dataroot data/nuscenes \
  --output-dir outputs/milestone4/full \
  --device mps \
  --seed 20261006
```

Each run writes `tracking_results.csv`, `metrics.json`, representative clean tracking overlays, and
`tracking_vs_severity.png`. The CSV contains scene/frame, corruption/severity/seed, predicted track
IDs, nuScenes instance tokens, boxes, IoU, match status, and linked predicted/ground-truth identity.
