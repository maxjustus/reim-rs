"""Target-lock post-filter for any pitch track (prototype).

The target is the voice that is clearest over time: its level sits near the
top of the recording's voiced-frame levels, and its pitch stays inside one
speaker's range. The lock starts a voiced run only when the frame's level is
within `level_db` of that target level and its pitch is within `range_cents`
of the target's median pitch. Everything else is reported unvoiced.

causal  the target level and pitch come from the last HISTORY_S of voiced
        frames: the pitch is the median of the frames within CLEAR_DB of the
        target level, so a louder voice takes over once it has spoken for a
        while. An accepted run continues while the pitch moves less than
        CONTINUE_CENTS per frame, so soft syllable tails stay voiced. No
        latency beyond the 30 ms level window.
offline the target level and pitch come from the whole recording (two passes),
        and each voiced run is accepted or rejected as a unit by its louder
        frames.
"""

from collections import deque

import numpy as np

LEVEL_WINDOW_S = 0.030
HISTORY_S = 2.0  # voiced-frame history for the causal target level and pitch
LEVEL_PERCENTILE = 90
CLEAR_DB = 3.0  # frames this close to the target level define the target pitch
MIN_PROFILE_FRAMES = 20  # clear frames before the pitch-range test applies
RUN_JUMP_CENTS = 300  # a larger frame-to-frame jump starts a new voiced run
CONTINUE_CENTS = 100  # per-frame pitch change that an accepted run may follow


def frame_levels(x, fs, times):
    """Level in dB of the LEVEL_WINDOW_S around each time."""
    w = int(LEVEL_WINDOW_S * fs)
    power = np.convolve(x**2, np.ones(w) / w, mode="same")
    i = np.clip(np.round(np.asarray(times) * fs).astype(int), 0, len(x) - 1)
    return 10 * np.log10(power[i] + 1e-12)


def lock_causal(times, f0, levels, level_db, range_cents):
    f0 = np.asarray(f0, dtype=float)
    out = np.zeros_like(f0)
    hop = np.median(np.diff(times)) if len(times) > 1 else 0.005
    history = int(HISTORY_S / hop)
    recent_levels = deque(maxlen=history)
    recent_cents = deque(maxlen=history)
    for i in np.flatnonzero(f0 > 0):
        cents = 1200 * np.log2(f0[i] / 440)
        recent_levels.append(levels[i])
        recent_cents.append(cents)
        level_array = np.asarray(recent_levels)
        target_level = np.percentile(level_array, LEVEL_PERCENTILE)
        clear = np.asarray(recent_cents)[level_array >= target_level - CLEAR_DB]
        continuing = i > 0 and out[i - 1] > 0 and abs(cents - 1200 * np.log2(out[i - 1] / 440)) <= CONTINUE_CENTS
        loud = levels[i] >= target_level - level_db
        in_range = len(clear) < MIN_PROFILE_FRAMES or abs(cents - np.median(clear)) <= range_cents
        if continuing or (loud and in_range):
            out[i] = f0[i]
    return out


def lock_offline(f0, levels, level_db, range_cents):
    f0 = np.asarray(f0, dtype=float)
    voiced = f0 > 0
    out = np.zeros_like(f0)
    if not voiced.any():
        return out
    cents = np.where(voiced, 1200 * np.log2(np.where(voiced, f0, 440) / 440), 0.0)
    target_level = np.percentile(levels[voiced], LEVEL_PERCENTILE)
    clear = voiced & (levels >= target_level - level_db)
    if not clear.any():
        return out
    center = np.median(cents[clear])
    for run in voiced_runs(voiced, cents):
        if (
            np.percentile(levels[run], LEVEL_PERCENTILE) >= target_level - level_db
            and abs(np.median(cents[run]) - center) <= range_cents
        ):
            out[run] = f0[run]
    return out


def voiced_runs(voiced, cents):
    """Index arrays of contiguous voiced frames, split at large pitch jumps."""
    runs, start = [], None
    for i, v in enumerate(voiced):
        if v and start is not None and abs(cents[i] - cents[i - 1]) > RUN_JUMP_CENTS:
            runs.append(np.arange(start, i))
            start = i
        elif v and start is None:
            start = i
        elif not v and start is not None:
            runs.append(np.arange(start, i))
            start = None
    if start is not None:
        runs.append(np.arange(start, len(voiced)))
    return runs
