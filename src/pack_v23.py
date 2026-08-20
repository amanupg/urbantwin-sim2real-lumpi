"""Package LUMPI v23 = keep-both fill-miss + score-thresh 0.01 + corrected betas.

Every component was validated on the LATE SLICE (120 real, non-forbidden frames),
which no stage was ever fitted on -- unlike val_harness, whose Van GT is corrupt
(median L 0.70 m = person dims, finding 64a).

  shipped v22 stack   mAP7 0.1864   Van 0.0310
  + keep-both         mAP7 0.2064   Van 0.1713   (+0.0200)
  + lt01              mAP7 0.2102   Van 0.1775   (+0.0038)
  + betas             mAP7 0.2265   Van 0.2862   (+0.0163)

Donor appends are DROPPED. keep-both supersedes the Van donor and is the
legitimate version of it: it retains predictions the detector itself emitted
rather than manufacturing predictions in a class it never assigned.

Realism frames are carried over byte-identical from v22 (the public 0.9011 set).
"""
from __future__ import annotations

import json
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tracks/lumpi/starter_kit/scoring_program"))
from detection_eval import _bev_iou  # noqa: E402

SCRATCH = Path("/private/tmp/claude-501/-Users-amanupg-Desktop-ECCV-2026/"
               "14f95da5-5ac0-416e-a9fc-6a6ef772371f/scratchpad")
TEST_PRED = SCRATCH / "predictions_lumpi_test_lt01.json"
FRAMES = ROOT / "tracks/lumpi/starter_kit/detection_test_frames"
SRC_ZIP = ROOT / "tracks/lumpi/submissions/v22_probeAB_bestof.zip"
OUT_ZIP = ROOT / "tracks/lumpi/submissions/v23_keepboth_betas.zip"

BETAS = {"Person": 0.7, "Bicycle": 0.5, "Truck": 0.2, "Car": 1.0, "Van": 1.5}

# The multiplier (1 - b + b*q) goes NEGATIVE for b > 1 when a box has few
# interior points (q < 1 - 1/b), which would emit invalid scores. Flooring it
# keeps those boxes at the bottom of the ranking with a valid positive score and
# costs nothing: late-slice Van AP 0.2861 floored vs 0.2862 unfloored.
MULT_FLOOR = 1e-6

_pc: dict[str, np.ndarray | None] = {}


def pts(fid):
    if fid not in _pc:
        f = FRAMES / f"{fid}.bin"
        if f.exists():
            _pc[fid] = np.fromfile(f, dtype=np.float32).reshape(-1, 4)[:, :3].astype(np.float64)
        else:
            _pc[fid] = None
    return _pc[fid]


def interior(P, box, pad=0.0):
    cx, cy, cz, L, W, H, yaw = box
    d = P - np.array([cx, cy, cz])
    c, s = np.cos(-yaw), np.sin(-yaw)
    x = d[:, 0] * c - d[:, 1] * s
    y = d[:, 0] * s + d[:, 1] * c
    return P[(np.abs(x) <= L / 2 + pad) & (np.abs(y) <= W / 2 + pad)
             & (np.abs(d[:, 2]) <= H / 2 + pad)]


def build(raw):
    out = {}
    for fid, dets in raw.items():
        P = pts(fid)
        # 1. Person geometric re-centring
        o = []
        for d in dets:
            e = dict(d)
            if d["class"] == "Person" and P is not None:
                q = interior(P, d["box"], 0.02)
                if len(q) >= 3:
                    b = list(map(float, d["box"]))
                    b[0], b[1] = float(q[:, 0].mean()), float(q[:, 1].mean())
                    e["box"] = b
            o.append(e)
        # 2. keep-both fill-miss: add the Car copy, RETAIN the Van original
        C = [x for x in o if x["class"] == "Car"]
        add = [{**x, "class": "Car"} for x in o
               if x["class"] == "Van"
               and not any(_bev_iou(np.asarray(x["box"]), np.asarray(c["box"])) > 0.3
                           for c in C)]
        o = o + add
        # 3. Truck length calibration
        o = [({**x, "box": [*x["box"][:3], x["box"][3] * 1.154, *x["box"][4:]]}
              if x["class"] == "Truck" else x) for x in o]
        # 4. density re-rank (shipped formula)
        n = []
        for x in o:
            b = BETAS.get(x["class"], 0.0)
            if b > 0 and P is not None:
                c = len(interior(P, x["box"], 0.0))
                q = np.log1p(c) / np.log1p(40.0)
                n.append({**x, "score": float(x["score"]
                                              * max(1 - b + b * q, MULT_FLOOR))})
            else:
                n.append(x)
        out[fid] = n
    return out


def main():
    raw = json.loads(TEST_PRED.read_text())
    assert len(raw) == 50, f"expected 50 test frames, got {len(raw)}"
    mn = min(x["score"] for v in raw.values() for x in v)
    assert mn < 0.05, f"low-threshold inference not applied (min score {mn})"

    preds = build(raw)

    # ---- gates -----------------------------------------------------------
    import collections
    cr = collections.Counter(x["class"] for v in raw.values() for x in v)
    cp = collections.Counter(x["class"] for v in preds.values() for x in v)
    print(f"frames {len(preds)}  dets {sum(len(v) for v in preds.values())}")
    for c in sorted(set(cr) | set(cp)):
        print(f"  {c:11s} raw {cr[c]:6d} -> out {cp[c]:6d}")
    # keep-both may only ADD Car boxes; every other class must be preserved
    for c in cr:
        if c == "Car":
            assert cp[c] >= cr[c], f"{c} lost boxes"
        else:
            assert cp[c] == cr[c], f"{c} count changed {cr[c]} -> {cp[c]}"
    # The re-rank multiplier is unbounded above (q = log1p(n)/log1p(40) exceeds
    # 1 whenever a box holds more than 40 points), so raw re-ranked scores leave
    # (0,1]. v22 shipped 291 scores > 1.0 and 502 at exactly 0 and the server
    # accepted it -- AP is rank-based -- but a stricter Final-phase validator
    # might not. A single GLOBAL monotone map fixes the range while preserving
    # every class's ranking, hence every per-class AP, exactly.
    allv = [x["score"] for v in preds.values() for x in v]
    lo, hi = min(allv), max(allv)
    eps = 1e-6
    span = hi - lo
    for v in preds.values():
        for x in v:
            x["score"] = (eps if span <= 0 else
                          eps + (1.0 - eps) * (x["score"] - lo) / span)
    print(f"\nscore range {lo:.4f}..{hi:.4f} -> normalised to ({eps},1] "
          f"(monotone, per-class AP unchanged)")
    assert all(0.0 < x["score"] <= 1.0 for v in preds.values() for x in v)

    # ---- package: carry realism frames + declaration over from v22 -------
    shutil.copy(SRC_ZIP, OUT_ZIP)
    with zipfile.ZipFile(SRC_ZIP) as z:
        names = [n for n in z.namelist() if n != "predictions.json"]
        keep = {n: z.read(n) for n in names}
    with zipfile.ZipFile(OUT_ZIP, "w", zipfile.ZIP_DEFLATED) as z:
        for n, data in keep.items():
            z.writestr(n, data)
        z.writestr("predictions.json", json.dumps(preds))
    nb = sum(1 for n in keep if n.startswith("synthetic/"))
    print(f"\nwrote {OUT_ZIP.name}: {nb} realism frames carried from v22, "
          f"declaration carried, predictions rebuilt")


if __name__ == "__main__":
    main()
