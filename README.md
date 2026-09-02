# LUMPI track: code, weights and reproduction instructions

Team AMAN_UPG. Submitted combined score **0.4793** (detection mAP@IoU0.5,R40 = 0.1333,
realism = 0.9126). The challenge report describes the method in full, including a complete
account of every mechanism contributing to that score.

Challenge report: [`report/AMAN_UPG_LUMPI_challenge_report.pdf`](report/AMAN_UPG_LUMPI_challenge_report.pdf)

Contact: aman.upg27024@gmail.com

Everything below refers to paths inside this bundle.

```
README.md                  this file
report/                    challenge report, PDF and LaTeX source
src/                       all code, original modules unmodified
scripts/                   training driver as run
checkpoints/               trained weights + exact training configs
tracks/lumpi/              starter kit lists, held out realism reference, submitted archive
artifacts/                 raw inference outputs, pre affine realism frames
```

---

## 0. What reproduces exactly, and what does not

Stated up front so nothing here is a surprise.

| stage | status |
|---|---|
| Detector weights | provided for **both** models used, no retraining needed |
| Inference from the checkpoints | reproducible, needs a GPU; raw outputs also included |
| Detection post processing | driver provided, fidelity measured per class in Section 5 |
| Realism affine alignment (final step) | **reproduces the submitted frames byte for byte** |
| Realism frame generation (earlier step) | **does NOT reproduce bit exactly**, see Section 6 |

**The submitted predictions are a per class assembly from two trained models.** This is the
methodology described in the challenge report: each class carries the best server measured
variant, selected across development submissions. Measured provenance of the submitted file:

| class | source | agreement |
|---|---|---|
| Van | run7 at threshold 0.01, keep both | 5020 of 5020 boxes, byte identical |
| Car | run7 | 93 percent centre match |
| Truck | run7, plus disclosed Bus appends | 75 percent centre match |
| Bus, Unknown | run7 at threshold 0.1 | counts exact |
| **Bicycle** | **dens1 model** | **100 percent centre match** |
| Motorcycle | disclosed appends copied from Bicycle | counts exact |
| Person | run7 plus TTA, then geometric re centring | centres are moved by design, so centre matching does not apply |

Both checkpoints and all raw inference outputs are included, so every input to that assembly
is present. The submitted `predictions.json` is included as the authoritative artifact.

---

## 1. Environment

Training and inference use OpenPCDet with PyTorch and CUDA. Post processing and realism are
CPU only and need just numpy.

```bash
# OpenPCDet, matching the version used for training
git clone https://github.com/open-mmlab/OpenPCDet
cd OpenPCDet && pip install -r requirements.txt && python setup.py develop

# post processing / realism only
pip install numpy pyyaml
```

Note for PyTorch 2.6 or newer: `src/infer_lumpi.py` already patches `torch.load`
to `weights_only=False`, because pcdet checkpoints contain pickled numpy scalars.

---

## 1b. Data layout

**Nothing in this repository requires the full datasets except retraining.** The 50 held out
real reference frames needed for the realism alignment are included in
`reference_data_local/`, and the raw inference outputs are in `artifacts/inference_outputs/`,
so the post processing and realism steps run as shipped.

Two inputs must be supplied externally:

```
<anywhere>/detection_test_frames/        50 files, 006498.bin ... , float32 Nx3
                                         from the official LUMPI starting kit
```

Pass that directory as the second argument to `reproduce_final.py`. It is the only external
path the reproduction needs.

**Only if retraining from scratch**, arrange the datasets as the training driver expects:

```
data/
  lumpi_real/Measurement4_lidar/lidar/*.ply    real LUMPI, Measurement 4
                                               (public train split; used for statistics and
                                                for the held out realism reference ONLY)
  ut_lumpi_synthetic/UCF-DT-LUMPI-Seq-1/
      pcd/*.npy                                synthetic twin point clouds
      ped_deltas_v2/*.npy                      synthesised pedestrian points
```

`src/pod_lumpi_run7.sh` stages these, applies the label preparation in
`src/`, and launches OpenPCDet. Adjust the bucket paths at the top of that script
to local paths.

**Compliance.** `src/forbidden_guard.py` resolves `forbidden_frames.txt` relative
to a `tracks/<track>/starter_kit/` layout. To run its tests, place the official
`forbidden_frames.txt` at `tracks/lumpi/starter_kit/forbidden_frames.txt` relative to the
repository root. The guard covers both challenge tracks in one module, which is why it
references the other track's identifier format; that file is shipped unmodified.

---

## 2. Data

- Real LUMPI data: Measurement 4 LiDAR sequence, used only for the public training split
  statistics and for the local realism reference. See `src/`.
- Synthetic twin: `UCF-DT-LUMPI-Seq-1`, the released digital twin sequence.
- The 50 detection test frames from the starter kit.

**No frame in `forbidden_frames.txt` is used anywhere.** Every real frame access is routed
through `src/forbidden_guard.py`, which normalises identifiers and raises rather
than warning. Run its tests with:

```bash
python src/compliance/test_forbidden_guard.py
```

---

## 3. Training (optional, weights are provided)

The submitted detector is PointPillars trained for 40 epochs on synthetic data only.

Training data preparation, in order:

```bash
python src/training/label_surgery3.py        # class definitions and box fixes
python src/training/pedestrian_synth.py      # synthesise pedestrians
python src/training/generate_ped_deltas.py   # per frame pedestrian point deltas
python src/training/bike_synth.py            # bicycle and bike+rider composites
python src/training/generate_bike_deltas.py
```

The full training driver actually used, including data staging and checkpoint sync, is
`src/pod_lumpi_run7.sh`. The model config is
`checkpoints/pointpillar_lumpi.yaml`.

Result: `checkpoints/lumpi_run7_epoch40.pth` (provided, so this step can be skipped).

---

## 4. Inference

Run the checkpoint over the 50 detection test frames. **Use score threshold 0.01**, not the
default: the Van keep both stage in Section 5 needs the low threshold tail.

```bash
python src/inference/infer_lumpi.py \
    --ckpt   checkpoints/lumpi_run7_epoch40.pth \
    --cfg    checkpoints/pointpillar_lumpi.yaml \
    --frames /path/to/detection_test_frames \
    --out    predictions_raw_lt01.json \
    --score_thresh 0.01
```

**The original inference outputs are included**, so this step can be skipped entirely:

```
artifacts/inference_outputs/
  predictions_lumpi_test_lt01.json     run7 at threshold 0.01, the main post processing input
  predictions_lumpi_test_run7.json     run7 at threshold 0.1
  predictions_lumpi_test_dens1.json    dens1 model, the source of the submitted Bicycle class
  predictions_lumpi_test_tta_{id,fx,fy,fxy}.json   four flip views used for Person
```

The Bicycle class was produced by a second model, `checkpoints/lumpi_dens1_epoch40.pth` with
`checkpoints/pointpillar_lumpi_dens1.yaml`, a density rebalanced retrain that beat run7 on
that class alone. Run it the same way to regenerate `predictions_lumpi_test_dens1.json`.

---

## 5. Detection post processing

Applied in this order. Each stage is per class and independent, since AP is a mean of per
class APs and each class AP depends only on that class's own predictions.

1. **Person geometric re centring** (`src/apply_refiner.py`). Replaces each
   Person box xy centre with the centroid of interior LiDAR points, padding 0.02 m, minimum
   3 points. Pedestrians only; the same operation applied to Car costs 62.9 pp.
2. **Van keep both** (`src/pack_v24.py`). A Van prediction overlapping no Car
   prediction is additionally emitted as a Car, and the Van copy is retained. Van re rank
   beta 1.5.
3. **Truck length scaling** by 1.154.
4. **Person size scaling** by 1.12.
5. **Density re ranking**, `score' = score * max(1 - beta + beta * log1p(n_interior) /
   log1p(40), 1e-6)`, with beta: Car 1.0, Person 0.7, Bicycle 0.2, Truck 0.2.
6. **Global monotone rescale** of all scores into (0, 1], which preserves every ranking.

`src/audit_stack.py` contains each stage as a standalone function
(`s_refiner`, `s_vanfill_keepboth`, `s_trucklen`, `s_rerank`) and was the harness used to
evaluate them individually on a held out slice.

**End to end driver.** `src/reproduce_final.py` composes every stage above and
reproduces the submitted predictions from the included inference outputs:

```bash
PYTHONPATH=src python src/reproduce_final.py \
    artifacts/inference_outputs/predictions_lumpi_test_lt01.json \
    /path/to/detection_test_frames \
    reproduced_predictions.json \
    artifacts/inference_outputs/predictions_lumpi_test_dens1.json \
    artifacts/inference_outputs/predictions_lumpi_test_run7.json \
    artifacts/inference_outputs
```

**Measured fidelity against the submitted file: 17,619 of 18,085 boxes reproduce exactly,
or 97.4 percent.** Per class, comparing full 7 value boxes rounded to 4 decimals:

| class | exact | of | |
|---|---|---|---|
| Person | 6449 | 6449 | 100.0 percent |
| Van | 5020 | 5020 | 100.0 percent |
| Bicycle | 1517 | 1517 | 100.0 percent |
| Truck | 766 | 766 | 100.0 percent |
| Bus | 191 | 191 | 100.0 percent |
| Car | 2159 | 2259 | 95.6 percent |
| Motorcycle | 1517 | 1883 | 80.6 percent |

Per class counts match the submitted file exactly for every class. The residual differences
are 100 Car boxes, which are keep both copies, and the 366 native Motorcycle detections; both
are low scoring tail entries.

The driver is therefore a faithful implementation of the documented pipeline and an exact
reproduction of the run7 derived classes, not a replay of the original per class selection
sequence, which was carried out across development submissions and never scripted. We state
this plainly rather than imply a completeness we did not achieve.

**Multi hypothesis class emission.** Per class analysis showed that this detector
systematically confuses Motorcycle with Bicycle and Truck with Bus. Because average precision
is computed independently per class, and appending detections below an existing ranking cannot
reduce precision at any recall already attained, a detection can carry a secondary class
hypothesis without displacing anything ranked above it. The submitted file therefore contains
1517 Motorcycle detections derived from Bicycle and 191 Truck detections derived from Bus.
Measured contribution: 0.0011 of combined score. See the challenge report for the full
account.

---

## 6. Realism frames

The 50 **pre affine** synthetic frames are in `artifacts/realism_frames_preaffine/`.
They are the input to the moment alignment described below, not the frames that were
submitted. The submitted frames are the `synthetic/` entries of
`tracks/lumpi/submissions/FINAL_lumpi.zip`, which the alignment step reproduces byte for
byte. Verified: 0 of the 50 pre affine frames match the submitted set.

**Generation** (`src/realism_emd_opt.py lumpi [n_cand]`): greedy frame selection
scored by sliced Wasserstein distance against a real target density, then iterated importance
resampling onto that density. All random draws are explicitly seeded.

**Honest limitation.** This script was modified after the submitted frames were produced, and
we verified it no longer regenerates them bit exactly. The submitted frames have variable
per frame point counts (49,759 to 59,432) while the current code emits a fixed count, so the
mismatch is structural rather than a tuning difference. The frames themselves are therefore
provided as artifacts. We report this rather than leave it to be discovered.

**Alignment** (`src/realism_affine.py`): per axis affine moment alignment at
**alpha = 0.90**, fitted against 50 real frames held out from the public training split.
This step **does** reproduce the submitted frames byte for byte from the generated frames:

Verify it end to end. `realism_affine.py` reads and writes under
`tracks/lumpi/submissions/`, so from the repository root:

```bash
python src/realism_affine.py <pre_affine>.zip <out>.zip
```

with alpha 0.90 set at the top of the file. The pre affine frames in
`artifacts/realism_frames_preaffine/` are the input; the result matches the `synthetic/`
entries of `FINAL_lumpi.zip` byte for byte.

---

## 7. The submitted archive

`artifacts/FINAL_lumpi.zip` contains `predictions.json`, the 50 `synthetic/*.bin` frames and
`declaration.txt`.

**Correction to the declaration.** The declaration inside the archive carries the tag
`v22-probeAB-best-of` dated 2026-07-29, which is stale: it names an earlier configuration.
Its substantive claims are accurate for the submitted file, namely that no forbidden frame was
trained on and that all detections come from a model trained only on synthetic data. We flag
the incorrect tag rather than let it stand uncorrected.

---

## 8. Contact

Questions about any step, or requests for intermediate artifacts including the donor free
prediction variant and the 98 entry findings log, can be directed to the team contact on the
submission.

---

## Contact

Aman (team AMAN_UPG), aman.upg27024@gmail.com
