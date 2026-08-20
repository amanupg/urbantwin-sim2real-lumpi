"""Forbidden-frame guard: the first data-layer boundary for both challenge tracks.

Every dataset loader, sampler, or adaptation step that touches REAL frames must
route frame IDs through this module before use. Violating the forbidden list is
grounds for disqualification, so failures here raise immediately rather than warn.

Canonical ID formats (from the starter kits):
    LUMPI:    zero-padded 6-digit index, e.g. "006498"
    V2X-Real: "<scenario>_<6-digit frame>", e.g. "10_000043"

The guard is deliberately strict about types and formats: a frame index that
arrives as int 6498 must be normalized to "006498" before lookup, otherwise a
formatting mismatch could silently pass a forbidden frame. normalize() handles
this; assert_allowed() applies it.
"""
from __future__ import annotations

import re
from pathlib import Path

_TRACKS_DIR = Path(__file__).resolve().parent.parent / "tracks"

_FORBIDDEN_FILES = {
    "lumpi": _TRACKS_DIR / "lumpi" / "starter_kit" / "forbidden_frames.txt",
    "v2x_real": _TRACKS_DIR / "v2x_real" / "starter_kit" / "forbidden_frames.txt",
}

_ID_PATTERNS = {
    "lumpi": re.compile(r"^\d{6}$"),
    "v2x_real": re.compile(r"^\d+_\d{6}$"),
}

_EXPECTED_COUNT = 100

_cache: dict[str, frozenset[str]] = {}


class ForbiddenFrameError(RuntimeError):
    """A forbidden frame reached the data layer. This is a disqualification risk."""


def normalize(frame_id: str | int, track: str) -> str:
    """Normalize a frame identifier to the track's canonical string form.

    Raises ValueError if the result does not match the canonical pattern,
    so malformed IDs can never be waved through.
    """
    if track not in _ID_PATTERNS:
        raise ValueError(f"unknown track: {track!r}")
    if isinstance(frame_id, int):
        if track != "lumpi":
            raise ValueError(
                f"bare int frame id {frame_id} is ambiguous for track {track!r}; "
                "pass the full canonical string"
            )
        frame_id = f"{frame_id:06d}"
    frame_id = frame_id.strip()
    if not _ID_PATTERNS[track].match(frame_id):
        raise ValueError(
            f"frame id {frame_id!r} does not match canonical format for {track}"
        )
    return frame_id


def forbidden_ids(track: str) -> frozenset[str]:
    """The set of forbidden canonical frame IDs for a track (cached)."""
    if track not in _cache:
        path = _FORBIDDEN_FILES.get(track)
        if path is None:
            raise ValueError(f"unknown track: {track!r}")
        ids = []
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            ids.append(normalize(line, track))
        if len(ids) != _EXPECTED_COUNT:
            raise RuntimeError(
                f"{path} yielded {len(ids)} ids, expected {_EXPECTED_COUNT}; "
                "refusing to proceed with a possibly-corrupt forbidden list"
            )
        _cache[track] = frozenset(ids)
    return _cache[track]


def is_forbidden(frame_id: str | int, track: str) -> bool:
    return normalize(frame_id, track) in forbidden_ids(track)


def assert_allowed(frame_id: str | int, track: str) -> str:
    """Gate a single frame ID. Returns the normalized ID if allowed.

    Raises ForbiddenFrameError if the frame is on the forbidden list.
    """
    norm = normalize(frame_id, track)
    if norm in forbidden_ids(track):
        raise ForbiddenFrameError(
            f"frame {norm} (track {track}) is in forbidden_frames.txt and must "
            "never be used for training, validation, tuning, or GT inspection"
        )
    return norm


def filter_allowed(frame_ids, track: str) -> list[str]:
    """Filter an iterable of frame IDs down to allowed ones (normalized).

    Use this when building training/validation splits from real data; unlike
    assert_allowed it drops forbidden frames instead of raising, but it logs
    the count so silent shrinkage is visible.
    """
    ids = [normalize(f, track) for f in frame_ids]
    bad = [f for f in ids if f in forbidden_ids(track)]
    if bad:
        print(
            f"[forbidden_guard] dropped {len(bad)} forbidden frames from "
            f"{track} split: {bad[:5]}{'...' if len(bad) > 5 else ''}"
        )
    return [f for f in ids if f not in forbidden_ids(track)]
