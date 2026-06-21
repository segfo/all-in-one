import re
import warnings

from pathlib import Path
from typing import List, Optional

from .typings import ChordSegment

SNAP_WINDOW = 0.35
DOWNBEAT_BONUS = 0.12
ORDER_PENALTY = 1.0
SHORT_SEGMENT_PENALTY = 0.5
MIN_CHORD_DURATION = 0.30


def detect_chords(
  audio_path: Path,
  beats: List[float],
  downbeats: List[float],
) -> List[ChordSegment]:
  try:
    raw = _run_madmom(audio_path)
    for seg in raw:
      seg.label = normalize_chord(seg.label_raw)
    return snap_to_beats(raw, beats, downbeats)
  except Exception as e:
    warnings.warn(f"Chord detection failed for {audio_path}: {e}")
    return []


def _run_madmom(audio_path: Path) -> List[ChordSegment]:
  from madmom.features.chords import DeepChromaChordRecognitionProcessor
  proc = DeepChromaChordRecognitionProcessor()
  output = proc(str(audio_path))
  segments = []
  for row in output:
    start = float(row[0])
    end = float(row[1])
    label = str(row[2])
    segments.append(ChordSegment(
      start=start,
      end=end,
      label=label,
      label_raw=label,
    ))
  return segments


_NORM_RE = re.compile(r'^([A-G][#b]?m?).*$')


def normalize_chord(label_raw: str) -> str:
  if label_raw in ('N', 'X', ''):
    return label_raw
  m = _NORM_RE.match(label_raw)
  return m.group(1) if m else label_raw


def snap_to_beats(
  segments: List[ChordSegment],
  beats: List[float],
  downbeats: List[float],
) -> List[ChordSegment]:
  if not segments:
    return []

  snapped = []
  prev_end = None

  for i, seg in enumerate(segments):
    next_start = segments[i + 1].start if i + 1 < len(segments) else None

    new_start = _choose_boundary(
      t=seg.start,
      beats=beats,
      downbeats=downbeats,
      prev_boundary=prev_end,
      next_boundary=seg.end,
      side='start',
    )
    new_end = _choose_boundary(
      t=seg.end,
      beats=beats,
      downbeats=downbeats,
      prev_boundary=new_start,
      next_boundary=next_start,
      side='end',
    )

    snapped.append(ChordSegment(
      start=new_start,
      end=new_end,
      label=seg.label,
      label_raw=seg.label_raw,
      confidence=seg.confidence,
    ))
    prev_end = new_end

  snapped = _merge_short_segments(snapped)
  snapped = _merge_consecutive_same_label(snapped)
  return snapped


def _collect_candidates(
  t: float,
  beats: List[float],
  window: float = SNAP_WINDOW,
) -> List[float]:
  return [b for b in beats if abs(b - t) <= window]


def _choose_boundary(
  t: float,
  beats: List[float],
  downbeats: List[float],
  prev_boundary: Optional[float] = None,
  next_boundary: Optional[float] = None,
  side: str = 'start',
) -> float:
  candidates = _collect_candidates(t, beats)
  if not candidates:
    return t

  downbeat_set = set(downbeats)
  best = None
  best_score = float('inf')

  for c in candidates:
    score = abs(c - t)

    if c in downbeat_set:
      score -= DOWNBEAT_BONUS

    if side == 'start' and prev_boundary is not None and c < prev_boundary:
      score += ORDER_PENALTY

    if side == 'end' and next_boundary is not None and c > next_boundary:
      score += ORDER_PENALTY

    if side == 'start' and next_boundary is not None:
      if (next_boundary - c) < MIN_CHORD_DURATION:
        score += SHORT_SEGMENT_PENALTY
    if side == 'end' and prev_boundary is not None:
      if (c - prev_boundary) < MIN_CHORD_DURATION:
        score += SHORT_SEGMENT_PENALTY

    if score < best_score:
      best_score = score
      best = c

  return best if best is not None else t


def _merge_short_segments(
  segments: List[ChordSegment],
  min_duration: float = MIN_CHORD_DURATION,
) -> List[ChordSegment]:
  if not segments:
    return []

  changed = True
  while changed:
    changed = False
    result = []
    i = 0
    while i < len(segments):
      seg = segments[i]
      if (seg.end - seg.start) < min_duration:
        if i == 0 and len(segments) > 1:
          # 最初のセグメントは次に統合
          next_seg = segments[i + 1]
          segments[i + 1] = ChordSegment(
            start=seg.start,
            end=next_seg.end,
            label=next_seg.label,
            label_raw=next_seg.label_raw,
            confidence=next_seg.confidence,
          )
          changed = True
          i += 1
          continue
        elif i > 0:
          # 前のセグメントに統合
          prev = result[-1]
          result[-1] = ChordSegment(
            start=prev.start,
            end=seg.end,
            label=prev.label,
            label_raw=prev.label_raw,
            confidence=prev.confidence,
          )
          changed = True
          i += 1
          continue
      result.append(seg)
      i += 1
    segments = result

  return segments


def _merge_consecutive_same_label(
  segments: List[ChordSegment],
) -> List[ChordSegment]:
  if not segments:
    return []

  result = [segments[0]]
  for seg in segments[1:]:
    if seg.label == result[-1].label:
      result[-1] = ChordSegment(
        start=result[-1].start,
        end=seg.end,
        label=result[-1].label,
        label_raw=result[-1].label_raw,
        confidence=result[-1].confidence,
      )
    else:
      result.append(seg)

  return result
