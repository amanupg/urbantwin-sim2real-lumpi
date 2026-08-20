"""Sanity tests for the forbidden-frame guard. Run: .venv/bin/python -m pytest src/ -q
or directly: .venv/bin/python src/test_forbidden_guard.py"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import forbidden_guard as fg


def test_counts():
    assert len(fg.forbidden_ids("lumpi")) == 100
    assert len(fg.forbidden_ids("v2x_real")) == 100


def test_detection_test_ids_are_forbidden():
    # The forbidden list is the union of detection-test and realism-reference
    # frames, so every detection test ID must be forbidden.
    tracks_dir = Path(__file__).resolve().parent.parent / "tracks"
    for track in ("lumpi", "v2x_real"):
        ids = json.loads(
            (tracks_dir / track / "starter_kit" / "detection_test_frame_ids.json").read_text()
        )
        assert len(ids) == 50
        for fid in ids:
            assert fg.is_forbidden(fid, track), f"{fid} should be forbidden ({track})"


def test_normalize_int_lumpi():
    assert fg.normalize(6498, "lumpi") == "006498"
    assert fg.is_forbidden(6498, "lumpi")


def test_normalize_rejects_bad():
    for bad in ("6498", "abc", "12345678", ""):
        try:
            fg.normalize(bad, "lumpi")
        except ValueError:
            pass
        else:
            raise AssertionError(f"normalize accepted {bad!r}")
    try:
        fg.normalize(43, "v2x_real")
    except ValueError:
        pass
    else:
        raise AssertionError("bare int accepted for v2x_real")


def test_assert_allowed():
    try:
        fg.assert_allowed("006498", "lumpi")
    except fg.ForbiddenFrameError:
        pass
    else:
        raise AssertionError("forbidden frame passed assert_allowed")
    # A frame not on the list passes and returns normalized form.
    assert fg.assert_allowed(1, "lumpi") == "000001"


def test_filter_allowed():
    out = fg.filter_allowed(["006498", "000001"], "lumpi")
    assert out == ["000001"]


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("all forbidden_guard tests passed")
