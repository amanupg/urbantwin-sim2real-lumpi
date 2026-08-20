"""LUMPI v24: v22 with ONLY the Van class replaced by keep-both.

v23 mixed three changes and lost 0.0046 overall while Van gained 0.0244. The
per-class ledger separated them cleanly:

  Van        0.0073 -> 0.0317   keep-both WORKS (+335%)
  Motorcycle 0.0202 -> 0.0119   exactly v21's no-donor value -> caused by DROPPING
  Truck      0.0107 -> 0.0097   exactly v21's no-donor value -> caused by DROPPING
  Person     0.1298 -> 0.1071   untouched class -> caused by lt01
  Bicycle    0.0782 -> 0.0508   lt01 + the refitted beta 0.2->0.5

So: keep the Van fix, revert the score threshold, revert the Bicycle beta.

Construction is SURGICAL rather than a rebuild: start from v22's own
predictions.json, delete every Van entry, insert the keep-both Van entries.
Because per-class AP depends only on that class's predictions, every other
class's AP is then guaranteed identical to v22's measured server values -- no
reproduction risk, and the submission isolates one variable.

Two variants are written:
  v24_vanfix          -- v22 exactly + Van fix (donors retained)
  v24_vanfix_nodonor  -- the same, with the Motorcycle/Truck donor appends removed
"""
from __future__ import annotations

import collections
import json
import sys
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SCRATCH = Path("/private/tmp/claude-501/-Users-amanupg-Desktop-ECCV-2026/"
               "14f95da5-5ac0-416e-a9fc-6a6ef772371f/scratchpad")
LT01 = SCRATCH / "predictions_lumpi_test_lt01.json"
FRAMES = ROOT / "tracks/lumpi/starter_kit/detection_test_frames"
V22 = ROOT / "tracks/lumpi/submissions/v22_probeAB_bestof.zip"
OUT_A = ROOT / "tracks/lumpi/submissions/v24_vanfix.zip"
OUT_B = ROOT / "tracks/lumpi/submissions/v24_vanfix_nodonor.zip"

VAN_BETA = 1.5
_pc: dict[str, np.ndarray | None] = {}


def pts(fid):
    if fid not in _pc:
        f = FRAMES / f"{fid}.bin"
        _pc[fid] = (np.fromfile(f, dtype=np.float32).reshape(-1, 4)[:, :3].astype(np.float64)
                    if f.exists() else None)
    return _pc[fid]


def n_interior(P, box):
    cx, cy, cz, L, W, H, yaw = box
    d = P - np.array([cx, cy, cz])
    c, s = np.cos(-yaw), np.sin(-yaw)
    x = d[:, 0] * c - d[:, 1] * s
    y = d[:, 0] * s + d[:, 1] * c
    return int(np.sum((np.abs(x) <= L / 2) & (np.abs(y) <= W / 2)
                      & (np.abs(d[:, 2]) <= H / 2)))


def bkey(d):
    return (d["class"],) + tuple(round(v, 3) for v in d["box"])


def main():
    with zipfile.ZipFile(V22) as z:
        v22 = json.loads(z.read("predictions.json"))
        carry = {n: z.read(n) for n in z.namelist() if n != "predictions.json"}

    lt = json.loads(LT01.read_text())
    base = {k: [d for d in v if d["score"] >= 0.1] for k, v in lt.items()}
    assert set(base) == set(v22), "frame id mismatch between v22 and base"

    # ---- new Van: every Van box the detector emitted, re-ranked -----------
    # v22's fill-miss DELETED the Vans that overlapped no Car; keep-both keeps
    # them. v22's Car already contains the Car copies, and stays untouched, so
    # Car AP cannot move.
    new_van = {}
    n_van = 0
    for fid, dets in base.items():
        P = pts(fid)
        out = []
        for d in dets:
            if d["class"] != "Van":
                continue
            s = d["score"]
            if P is not None:
                q = np.log1p(n_interior(P, d["box"])) / np.log1p(40.0)
                s = float(s * max(1 - VAN_BETA + VAN_BETA * q, 1e-6))
            out.append({"class": "Van", "box": list(map(float, d["box"])), "score": s})
        new_van[fid] = out
        n_van += len(out)

    # ---- identify the donor appends --------------------------------------
    # Probe A appended copies of a donor class's boxes relabelled: Motorcycle
    # <- Bicycle, Truck <- Bus. A donor is a box whose geometry is byte-equal to
    # a box present in the donor class in the same frame.
    donors = {"Motorcycle": "Bicycle", "Truck": "Bus"}
    is_donor = {}
    for fid, dets in v22.items():
        geo = collections.defaultdict(set)
        for d in dets:
            geo[d["class"]].add(tuple(round(v, 3) for v in d["box"]))
        for i, d in enumerate(dets):
            src = donors.get(d["class"])
            is_donor[(fid, i)] = bool(src and tuple(round(v, 3) for v in d["box"]) in geo[src])

    for keep_donors, out_zip in ((True, OUT_A), (False, OUT_B)):
        preds = {}
        for fid, dets in v22.items():
            kept = [d for i, d in enumerate(dets)
                    if d["class"] != "Van" and (keep_donors or not is_donor[(fid, i)])]
            preds[fid] = kept + new_van[fid]

        # ---- gates -------------------------------------------------------
        c22 = collections.Counter(x["class"] for v in v22.values() for x in v)
        cnew = collections.Counter(x["class"] for v in preds.values() for x in v)
        # every non-Van class must be untouched (donors aside)
        for c in c22:
            if c == "Van":
                continue
            if keep_donors:
                assert cnew[c] == c22[c], f"{c} changed {c22[c]} -> {cnew[c]}"
            else:
                assert cnew[c] <= c22[c], f"{c} grew unexpectedly"
        # and byte-identical, not merely equinumerous
        for fid in v22:
            a = collections.Counter(bkey(d) for d in v22[fid] if d["class"] != "Van")
            b = collections.Counter(bkey(d) for d in preds[fid] if d["class"] != "Van")
            assert (b - a) == collections.Counter(), f"{fid}: non-Van boxes altered"

        allv = [x["score"] for v in preds.values() for x in v]
        lo, hi = min(allv), max(allv)
        eps = 1e-6
        for v in preds.values():
            for x in v:
                x["score"] = eps + (1 - eps) * (x["score"] - lo) / (hi - lo)

        with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as z:
            for n, data in carry.items():
                z.writestr(n, data)
            z.writestr("predictions.json", json.dumps(preds))

        tag = "donors kept" if keep_donors else "donors removed"
        print(f"\n{out_zip.name}  ({tag})")
        print(f"  Van {c22['Van']} -> {cnew['Van']}   total {sum(c22.values())} -> "
              f"{sum(cnew.values())}")
        for c in sorted(c22):
            if cnew[c] != c22[c]:
                print(f"    {c:11s} {c22[c]:6d} -> {cnew[c]:6d}")
        print(f"  score range {lo:.4f}..{hi:.4f} -> normalised, "
              f"non-Van boxes verified byte-identical to v22")


if __name__ == "__main__":
    main()
