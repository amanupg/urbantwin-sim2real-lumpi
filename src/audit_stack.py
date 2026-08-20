"""Audit the shipped post-processing stack, one stage at a time, on the late slice.

Every stage in the v18/v21/v22 stack was selected on val_harness (which inflates
2.5-21x per class and has corrupt Van GT) plus per-class server feedback across
~20 submissions. The late slice is 120 real, non-forbidden frames that no stage
was ever fitted on -- the first genuinely held-out check we have.

A stage that helps on val_harness and the server but HURTS here is fitted to
noise. A stage that hurts here and on the server is simply wrong.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tracks/lumpi/starter_kit/scoring_program"))
from detection_eval import evaluate_3d_detection, _bev_iou  # noqa: E402

sys.path.insert(0, str(ROOT / "src"))
from forbidden_guard import filter_allowed  # noqa: E402

LATE = ROOT / "tracks/lumpi/val_late"
CLASSES = ["Person", "Car", "Bicycle", "Motorcycle", "Bus", "Truck", "Van", "Unknown"]
SCORED = ["Person", "Car", "Bicycle", "Motorcycle", "Bus", "Truck", "Van"]

_pts_cache: dict[str, np.ndarray | None] = {}


def pts(fid):
    if fid not in _pts_cache:
        f = LATE / f"frames/{fid}.bin"
        if f.exists():
            a = np.fromfile(f, dtype=np.float32)
            # late-slice frames are xyzi
            _pts_cache[fid] = a.reshape(-1, 4)[:, :3].astype(np.float64)
        else:
            _pts_cache[fid] = None
    return _pts_cache[fid]


def interior(P, box, pad=0.0):
    cx, cy, cz, L, W, H, yaw = box
    d = P - np.array([cx, cy, cz])
    c, s = np.cos(-yaw), np.sin(-yaw)
    x = d[:, 0] * c - d[:, 1] * s
    y = d[:, 0] * s + d[:, 1] * c
    m = ((np.abs(x) <= L / 2 + pad) & (np.abs(y) <= W / 2 + pad)
         & (np.abs(d[:, 2]) <= H / 2 + pad))
    return P[m]


# ------------------------------------------------------------------- stages
def s_refiner(preds):
    """Person geometric re-centring (pad 0.02, minp 3)."""
    out = {}
    for fid, dets in preds.items():
        P = pts(fid)
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
        out[fid] = o
    return out


def s_vanfill(preds):
    """Van fill-miss: a Van prediction not overlapping any Car prediction
    becomes a Car."""
    out = {}
    for fid, dets in preds.items():
        C = [x for x in dets if x["class"] == "Car"]
        out[fid] = [({**x, "class": "Car"}
                     if (x["class"] == "Van"
                         and not any(_bev_iou(np.asarray(x["box"]),
                                              np.asarray(c["box"])) > 0.3 for c in C))
                     else x) for x in dets]
    return out


def s_trucklen(preds):
    """Truck length x1.154."""
    return {fid: [({**x, "box": [*x["box"][:3], x["box"][3] * 1.154, *x["box"][4:]]}
                   if x["class"] == "Truck" else x) for x in dets]
            for fid, dets in preds.items()}


def s_rerank(preds, betas):
    out = {}
    for fid, dets in preds.items():
        P = pts(fid)
        o = []
        for x in dets:
            b = betas.get(x["class"], 0.0)
            if b > 0 and P is not None:
                c = len(interior(P, x["box"], 0.0))
                q = np.log1p(c) / np.log1p(40.0)
                o.append({**x, "score": float(x["score"] * (1 - b + b * q))})
            else:
                o.append(x)
        out[fid] = o
    return out


V18_BETAS = {"Person": 0.7, "Bicycle": 0.2, "Truck": 0.2}
V22_BETAS = {"Person": 0.7, "Bicycle": 0.2, "Truck": 0.2, "Car": 1.0}


def ev(preds, gt):
    r = evaluate_3d_detection(preds, gt, CLASSES, iou_thresholds=(0.5, 0.7),
                              iou_type="3d")
    ap = {}
    for c in SCORED:
        v = r["per_class"].get(c, {}).get("AP_0.5_R40", float("nan"))
        ap[c] = 0.0 if v != v else float(v)
    return float(np.mean([ap[c] for c in SCORED])), ap


def show(tag, m, ap, base=None):
    cells = []
    for c in SCORED:
        s = f"{ap[c]:.4f}"
        if base is not None:
            d = ap[c] - base[c]
            if abs(d) >= 0.0005:
                s += f"({d:+.3f})"
        cells.append(f"{c[:4]} {s}")
    print(f"{tag:26s} mAP7={m:.4f}  " + "  ".join(cells))


def main():
    gt = json.loads((LATE / "detection_gt.json").read_text())
    pr = json.loads((LATE / "predictions_lumpi_late_run7.json").read_text())
    ids = filter_allowed(sorted(set(gt) & set(pr)), "lumpi")
    gt = {i: gt[i] for i in ids}
    pr = {i: pr[i] for i in ids}
    print(f"late slice: {len(ids)} real non-forbidden frames\n")

    m0, ap0 = ev(pr, gt)
    show("run7 raw", m0, ap0)

    print("\n--- each stage applied ALONE to raw ---")
    for name, fn in [("+ Person refiner", s_refiner),
                     ("+ Van fill-miss", s_vanfill),
                     ("+ Truck len x1.154", s_trucklen),
                     ("+ re-rank v18", lambda p: s_rerank(p, V18_BETAS)),
                     ("+ re-rank v22 (with Car)", lambda p: s_rerank(p, V22_BETAS))]:
        m, ap = ev(fn(pr), gt)
        show(name, m, ap, ap0)

    print("\n--- cumulative, in shipped order ---")
    cur, tag = pr, "run7 raw"
    for name, fn in [("refiner", s_refiner),
                     ("vanfill", s_vanfill),
                     ("trucklen", s_trucklen),
                     ("rerank v22", lambda p: s_rerank(p, V22_BETAS))]:
        cur = fn(cur)
        tag = tag + " +" + name
        m, ap = ev(cur, gt)
        show(tag[-26:], m, ap, ap0)

    print("\n--- shipped stack WITHOUT Van fill-miss ---")
    cur = s_rerank(s_trucklen(s_refiner(pr)), V22_BETAS)
    m, ap = ev(cur, gt)
    show("full minus vanfill", m, ap, ap0)


if __name__ == "__main__":
    main()


def s_vanfill_keepboth(preds):
    """fill-miss WITHOUT the deletion: a Van prediction that overlaps no Car
    prediction is ALSO emitted as a Car, but the Van copy is retained.

    AP is per-class and each class's AP depends only on its own predictions, so
    adding a Car copy cannot touch Van AP and keeping the Van copy cannot touch
    Car AP. The detector already emits boxes in multiple classes natively
    (MULTI_CLASSES_NMS: True); the shipped fill-miss rule was destroying that.
    """
    out = {}
    for fid, dets in preds.items():
        C = [x for x in dets if x["class"] == "Car"]
        o = list(dets)
        for x in dets:
            if x["class"] != "Van":
                continue
            if not any(_bev_iou(np.asarray(x["box"]), np.asarray(c["box"])) > 0.3
                       for c in C):
                o.append({**x, "class": "Car"})
        out[fid] = o
    return out


def main_keepboth():
    gt = json.loads((LATE / "detection_gt.json").read_text())
    pr = json.loads((LATE / "predictions_lumpi_late_run7.json").read_text())
    ids = filter_allowed(sorted(set(gt) & set(pr)), "lumpi")
    gt = {i: gt[i] for i in ids}
    pr = {i: pr[i] for i in ids}
    m0, ap0 = ev(pr, gt)
    show("run7 raw", m0, ap0)
    shipped = s_rerank(s_trucklen(s_vanfill(s_refiner(pr))), V22_BETAS)
    m, ap = ev(shipped, gt)
    show("SHIPPED (v22 stack)", m, ap, ap0)
    kb = s_rerank(s_trucklen(s_vanfill_keepboth(s_refiner(pr))), V22_BETAS)
    m2, ap2 = ev(kb, gt)
    show("keep-both fill-miss", m2, ap2, ap0)
    print(f"\nkeep-both vs shipped: mAP7 {m:.4f} -> {m2:.4f}  ({m2 - m:+.4f})")
    for c in SCORED:
        d = ap2[c] - ap[c]
        if abs(d) >= 0.0005:
            print(f"  {c:11s} {ap[c]:.4f} -> {ap2[c]:.4f}  ({d:+.4f})")
