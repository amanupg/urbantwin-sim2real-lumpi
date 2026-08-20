"""Post-process: recentre Person boxes on the centroid of their interior points.

Pure geometry -- uses NO real labels, so the detector pipeline stays
unambiguously sim-only (unlike the learned variant, which is trained on real
GT boxes and sits in a compliance grey zone until the organizers clarify).

Measured cross-slice (train-derived on early harness, tested on late slice,
run5 predictions, 2026-07-27 Opus agent):
    mAP@0.5   0.1512 -> 0.1609   (+0.0097)
    Person AP 0.0332 -> 0.1011   (3.0x)
Learned variant for reference: mAP 0.1631, Person 0.1159 -- 18% more gain,
but needs organizer clearance.

Why it works: a pedestrian is small and roughly convex, so LiDAR returns
surround them and the interior centroid IS essentially the true centre. The
detector cannot match that precision through a 0.3125 m BEV cell. It is applied
to Person ONLY: a car is seen only on the faces pointing at the sensor, so its
visible centroid sits ~0.8 m off the true centre (measured: applying this to Car
costs -62.9pp pass@0.5).

Usage:
  .venv/bin/python src/apply_refiner.py <predictions.json> <frames_dir> <out.json>
"""
from __future__ import annotations

import json
import sys
import numpy as np
from pathlib import Path

REFINE_CLASSES = {"Person"}
MIN_PTS = 5
PAD = 0.15


def interior(pts, box, pad=PAD):
    cx, cy, cz, l, w, h, hd = box[:7]
    q = pts - np.array([cx, cy, cz])
    c, s = np.cos(-hd), np.sin(-hd)
    x = q[:, 0] * c - q[:, 1] * s
    y = q[:, 0] * s + q[:, 1] * c
    m = ((np.abs(x) <= l / 2 + pad) & (np.abs(y) <= w / 2 + pad)
         & (np.abs(q[:, 2]) <= h / 2 + pad))
    return pts[m]


def main(pred_path, frames_dir, out_path):
    preds = json.loads(Path(pred_path).read_text())
    fdir = Path(frames_dir)
    out, n_ref, n_skip, shifts = {}, 0, 0, []
    for fid, dets in preds.items():
        f = fdir / f"{fid}.bin"
        pts = (np.fromfile(f, dtype=np.float32).reshape(-1, 3).astype(np.float64)
               if f.exists() else None)
        new = []
        for d in dets:
            d2 = dict(d)
            if d.get("class") in REFINE_CLASSES and pts is not None:
                box = np.array(d["box"], dtype=np.float64)
                q = interior(pts, box)
                if len(q) >= MIN_PTS:
                    b = list(map(float, d["box"]))
                    nx, ny = float(q[:, 0].mean()), float(q[:, 1].mean())
                    shifts.append(np.hypot(nx - b[0], ny - b[1]))
                    b[0], b[1] = nx, ny
                    d2["box"] = b
                    n_ref += 1
                else:
                    n_skip += 1
            new.append(d2)
        out[fid] = new
    Path(out_path).write_text(json.dumps(out))
    tot = sum(len(v) for v in preds.values())
    print(f"frames {len(out)}  total dets {tot}  Person refined {n_ref}  "
          f"skipped(<{MIN_PTS} pts) {n_skip}")
    if shifts:
        s = np.array(shifts)
        print(f"  centre shift: median {np.median(s):.3f} m, p90 {np.percentile(s,90):.3f} m")
    # integrity: nothing but Person xy may change
    for fid in preds:
        for a, b in zip(preds[fid], out[fid]):
            assert a["class"] == b["class"] and a["score"] == b["score"]
            if a["class"] not in REFINE_CLASSES:
                assert a["box"] == b["box"], "non-Person box modified"
            else:
                assert list(a["box"])[2:] == list(b["box"])[2:], "Person z/dims/yaw modified"
    print("  integrity check passed: only Person xy changed")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise SystemExit(__doc__)
    main(*sys.argv[1:])
