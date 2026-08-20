#!/bin/bash
# Pod driver: LUMPI run7 = run5 proven base + ONE change: unmerge Truck/Bus.
# Drops articulated (server-neutral, didn't earn its place).
# Everything else is run5: surgery3, 3-mode Truck anchors, bike composites, ped v2 groups.
set -u
B=gs://eccv-data-bucket-amlc
WORK=/workspace
export PATH="$HOME/google-cloud-sdk/bin:$PATH"
source "$WORK/venv/bin/activate"
export TORCH_FORCE_NO_WEIGHTS_LOAD=1
PCD=$WORK/OpenPCDet_lumpi
mark() { for i in 1 2 3; do echo "$2 @ $(date -u)" | gcloud storage cp - "$B/status/$1.txt" 2>/dev/null && break || sleep 5; done; }
last_ckpt() { ls -t "$PCD"/output/custom_models/pointpillar_lumpi/run7*/ckpt/checkpoint_epoch_*.pth 2>/dev/null | head -1 || true; }

if [ ! -f "$PCD/data/custom/.run7_ready" ]; then
  cd "$WORK"
  # 1. Download surgery3 + ped v3 (reduced density: 12/frame vs 22) + bike deltas (NO art_deltas)
  for t in lbl_surgery3 ped_deltas_v3 bike_deltas; do
    [ -d $t ] || { gcloud storage cp $B/insomnia/$t.tgz . && tar xzf $t.tgz; }
    [ -d $t ] || { gcloud storage cp $B/$t.tgz . && tar xzf $t.tgz; }
    find $t -name '._*' -delete 2>/dev/null || true
  done
  # 1b. If ped_deltas_v3 not available, fall back to ped_deltas (v2 groups, 22/frame)
  [ -d ped_deltas_v3 ] || { cp -r ped_deltas ped_deltas_v3 2>/dev/null; echo "WARNING: using ped_deltas (full density)"; }
  # 2. Labels: surgery3 + ped v3 + bike, KEEP Truck and Bus SEPARATE (no merge), 8-field
  python - <<'PY'
from pathlib import Path
work = Path('/workspace')
lbl_dir = work/'OpenPCDet_lumpi/data/custom/labels'
n=0
for lp in sorted((work/'lbl_surgery3').glob('*.txt')):
    rows = lp.read_text().splitlines()
    for extra_dir in ('ped_deltas_v3','bike_deltas'):
        dl = work/extra_dir/lp.name
        if dl.exists():
            rows += [r for r in dl.read_text().splitlines() if r.strip()]
    out=[]
    for r in rows:
        f = r.split()
        if len(f)<8: continue
        # NO MERGE — keep original class names (Truck stays Truck, Bus stays Bus)
        out.append(' '.join(f[:8]))
    (lbl_dir/lp.name).write_text('\n'.join(out)+('\n' if out else ''))
    n+=1
print('labels rewritten (NO merge, ped v3):',n)
PY
  # 3. Config: run5 base with Truck/Bus UNMERGED
  #    Truck: 3-mode anchors [8.45, 12.0, 16.5] (from run5, proven)
  #    Bus: single anchor [12.0, 2.90, 3.30] (Kimi's suggestion)
  python - <<'PY'
from pathlib import Path
cfg = Path('/workspace/OpenPCDet_lumpi/tools/cfgs/custom_models/pointpillar_lumpi.yaml')
s = cfg.read_text()
# Replace CLASS_NAMES: add Bus back
s = s.replace("CLASS_NAMES: ['Car', 'Van', 'TruckBus', 'Bicycle', 'Motorcycle', 'Unknown', 'Person']",
              "CLASS_NAMES: ['Car', 'Van', 'Truck', 'Bus', 'Bicycle', 'Motorcycle', 'Unknown', 'Person']")
# Replace TruckBus anchor with Truck (3-mode) + Bus (single)
s = s.replace(
    "{\n                'class_name': 'TruckBus',\n                'anchor_sizes': [[8.45, 2.89, 3.33], [12.0, 2.90, 3.30], [16.5, 3.50, 4.20]],",
    "{\n                'class_name': 'Truck',\n                'anchor_sizes': [[8.45, 2.89, 3.33], [12.0, 2.90, 3.30], [16.5, 3.50, 4.20]],"
)
# Add Bus anchor after Truck block (before Bicycle)
bus_block = """            {
                'class_name': 'Bus',
                'anchor_sizes': [[12.0, 2.90, 3.30]],
                'anchor_rotations': [0, 1.57],
                'anchor_bottom_heights': [-2.15],
                'align_center': False,
                'feature_map_stride': 2,
                'matched_threshold': 0.55,
                'unmatched_threshold': 0.4
            },
"""
s = s.replace("            {\n                'class_name': 'Bicycle',", bus_block + "            {\n                'class_name': 'Bicycle',")
# Replace RPN_HEAD_CFGS
s = s.replace(
    "{ 'HEAD_CLS_NAME': ['TruckBus'] },",
    "{ 'HEAD_CLS_NAME': ['Truck'] },\n            { 'HEAD_CLS_NAME': ['Bus'] },"
)
cfg.write_text(s)
print('config patched: Truck/Bus unmerged, 8 classes')
PY
  # 4. Points: bike deltas merged in place (NO art_deltas)
  python - <<'PY'
import numpy as np
from pathlib import Path
work = Path('/workspace')
pts_dir = work/'OpenPCDet_lumpi/data/custom/points'
marker = work/'OpenPCDet_lumpi/data/custom'/'.bike_merged'
if not marker.exists():
    n=0
    for f in sorted(pts_dir.glob('*.npy')):
        d = work/'bike_deltas'/f.name
        if not d.exists(): continue
        add = np.load(d)
        if len(add):
            base=np.load(f); np.save(f, np.concatenate([base, add.astype(base.dtype)])); n+=len(add)
    marker.write_text('done'); print('bike merged points:',n)
PY
  # 5. Infos
  cd "$PCD"
  rm -f data/custom/custom_infos_train.pkl data/custom/custom_infos_val.pkl
  rm -rf data/custom/gt_database
  python -m pcdet.datasets.custom.custom_dataset create_custom_infos tools/cfgs/dataset_configs/lumpi_ut.yaml
  ls data/custom/custom_infos_train.pkl || { echo INFOS_FAILED; exit 1; }
  touch "$PCD/data/custom/.run7_ready"
  echo SETUP_DONE
fi

cd "$PCD/tools"
gcloud storage rsync -r "$B/checkpoints" "$PCD/output" || true
setsid bash -c 'while true; do gcloud storage rsync -r "$PCD/output" "$B/checkpoints" >/dev/null 2>&1; sleep 300; done' >/dev/null 2>&1 &
mark lumpi_run7 "RUNNING (run7, Truck/Bus unmerged)"
latest=$(last_ckpt)
resume=""
[ -n "$latest" ] && resume="--ckpt $latest"
echo "launching train.py resume='$resume'"
python train.py --cfg_file cfgs/custom_models/pointpillar_lumpi.yaml --extra_tag run7 $resume > "$WORK/train_lumpi_run7.log" 2>&1
rc=$?
echo "train.py exited rc=$rc"
last=$(last_ckpt)
if [ -n "$last" ] && [[ "$last" == *"epoch_40.pth" ]]; then
  mark lumpi_run7 DONE
  mkdir -p "$WORK/frames"
  for d in lumpi_test_frames lumpi_val_frames lumpi_late_frames; do
    [ -d "$WORK/frames/$d" ] || gcloud storage cp -r "$B/$d" "$WORK/frames/"
  done
  V=$(last_ckpt)
  python "$PCD/infer_lumpi.py" --ckpt "$V" --cfg cfgs/custom_models/pointpillar_lumpi.yaml \
    --frames "$WORK/frames/lumpi_test_frames" --out "$WORK/predictions_lumpi_test_run7.json" --score_thresh 0.1
  python "$PCD/infer_lumpi.py" --ckpt "$V" --cfg cfgs/custom_models/pointpillar_lumpi.yaml \
    --frames "$WORK/frames/lumpi_val_frames" --out "$WORK/predictions_lumpi_val_run7.json" --score_thresh 0.1
  python "$PCD/infer_lumpi.py" --ckpt "$V" --cfg cfgs/custom_models/pointpillar_lumpi.yaml \
    --frames "$WORK/frames/lumpi_late_frames" --out "$WORK/predictions_lumpi_late_run7.json" --score_thresh 0.1
  gcloud storage cp "$WORK"/predictions_lumpi_*_run7.json "$B/preds/" || true
  gcloud storage rsync -r "$PCD/output" "$B/checkpoints" || true
  echo LUMPI_RUN7_COMPLETE
  exit 0
fi
CNT_FILE="$WORK/.lumpi_resubmit_count"
cnt=0; [ -f "$CNT_FILE" ] && cnt=$(cat "$CNT_FILE")
if [ "$rc" -ne 1 ] && [ "$cnt" -lt 3 ]; then
  echo $((cnt+1)) > "$CNT_FILE"
  nohup bash "$0" >/dev/null 2>&1 &
else
  mark lumpi_run7 "FAILED_RC${rc}_NOEPOCH40 (runpod)"
fi
