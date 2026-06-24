import re
import traceback
import warnings

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .typings import ChordSegment

STEM_WEIGHTS: Dict[str, float] = {
  "bass": 0.5,
  "other": 0.4,
  "vocals": 0.1,
  "mix": 1.0,
}

# stem 無音ゲート: 各区間で stem の RMS が自ピークの相対閾値未満なら投票から除外する。
# これにより bass 無音区間（イントロ等）では bass の "N" 票が捨てられ、other が和声を決める。
STEM_SILENCE_REL = 0.10        # 各 stem 自身のピーク RMS に対する相対無音閾値
STEM_SILENCE_ABS_FLOOR = 1e-4  # これ未満は常に無音（全体が無音な stem の相対比破綻を防ぐ）
RMS_FRAME_LENGTH = 2048
RMS_HOP_LENGTH = 512

# (frame_times, frame_rms, peak_rms)
StemActivity = Tuple[np.ndarray, np.ndarray, float]

SNAP_WINDOW = 0.35
DOWNBEAT_TOL = 0.20      # Phase1: downbeat強制スナップの判定半径（秒）
DOWNBEAT_BONUS = 0.12    # Phase2: downbeatのソフト優遇ボーナス
ORDER_PENALTY = 1.0
SHORT_SEGMENT_PENALTY = 0.5
MIN_CHORD_DURATION = 0.30
FLOAT_TOL = 1e-3


def detect_chords(
  audio_path: Path,
  beats: List[float],
  downbeats: List[float],
  demix_path: Optional[Path] = None,
) -> List[ChordSegment]:
  try:
    stem_results, activities = _run_madmom_multi(audio_path, demix_path)
    merged = _merge_stem_results(stem_results, activities)

    for seg in merged:
      seg.label = normalize_chord(seg.label_raw)

    snapped = snap_to_beats(merged, beats, downbeats)
    return snapped

  except Exception as e:
    import sys
    print(f"\n[chord_detection] FAILED: {e}", file=sys.stderr)
    traceback.print_exc(file=sys.stderr)
    return []


def _compute_stem_activity(path: Path) -> StemActivity:
  """stem 音声のフレーム RMS エンベロープとピーク RMS を返す。

  無音ゲート（区間ごとに「実際に音が鳴っているか」を判定）に用いる。
  戻り値は (frame_times, frame_rms, peak_rms)。
  """
  import librosa

  y, sr = librosa.load(str(path), sr=None, mono=True)
  frame_rms = librosa.feature.rms(
    y=y, frame_length=RMS_FRAME_LENGTH, hop_length=RMS_HOP_LENGTH,
  )[0]
  frame_times = librosa.frames_to_time(
    np.arange(len(frame_rms)), sr=sr, hop_length=RMS_HOP_LENGTH,
  )
  peak_rms = float(frame_rms.max()) if len(frame_rms) else 0.0
  return frame_times, frame_rms, peak_rms


def _stem_active(activity: StemActivity, t0: float, t1: float) -> bool:
  """区間 [t0, t1) で stem が鳴っているか（無音でないか）を判定する。

  - 区間内にフレームが無い（極短区間など）→ True（誤除外を避ける安全側）。
  - 閾値は stem 自身のピーク RMS に対する相対値（絶対フロア併用）。
  """
  frame_times, frame_rms, peak_rms = activity
  mask = (frame_times >= t0) & (frame_times < t1)
  if not mask.any():
    return True
  mean_rms = float(frame_rms[mask].mean())
  threshold = max(STEM_SILENCE_REL * peak_rms, STEM_SILENCE_ABS_FLOOR)
  return mean_rms >= threshold


def _run_madmom_multi(
  audio_path: Path,
  demix_path: Optional[Path],
) -> Tuple[Dict[str, List[ChordSegment]], Dict[str, StemActivity]]:
  # CNN ベースのコード認識器（CNNChordFeatureProcessor + CRFChordRecognitionProcessor）。
  # DeepChroma より N が少なく種別精度も高い（docs/02_chord_detection.md §9 の A/B 実測）。
  # CNN 特徴抽出器は音声ファイルを直接受け取る。
  from madmom.features.chords import (
    CNNChordFeatureProcessor,
    CRFChordRecognitionProcessor,
  )

  feat_proc = CNNChordFeatureProcessor()      # モデルロードを伴うのでループ外で1回だけ生成
  decode = CRFChordRecognitionProcessor()

  # stem ファイルを収集。
  stems: Dict[str, Path] = {}
  if demix_path is not None:
    for name in ("bass", "other", "vocals"):
      p = demix_path / f"{name}.wav"
      if p.exists():
        stems[name] = p
  # mix（オリジナル音源）を常に併走させる。
  # - stem が無ければ mix が単独の主役（従来のフォールバック）。
  # - stem がある場合、mix は primary 投票が N の区間を埋める fallback voter として使う
  #   （_merge_stem_results 参照）。分離 stem 単体では N でも、全和声が混ざった mix なら
  #   コードを取れる区間があるため。
  stems["mix"] = audio_path

  results: Dict[str, List[ChordSegment]] = {}
  activities: Dict[str, StemActivity] = {}
  for name, path in stems.items():
    output = decode(feat_proc(str(path)))

    segments = []
    for row in output:
      # madmom returns a numpy structured array; access by field name
      try:
        start = float(row['start'])
        end = float(row['end'])
        label = str(row['label'])
      except (KeyError, TypeError, ValueError):
        start, end, label = float(row[0]), float(row[1]), str(row[2])

      segments.append(ChordSegment(
        start=start,
        end=end,
        label=label,
        label_raw=label,
      ))

    results[name] = segments
    activities[name] = _compute_stem_activity(path)

  return results, activities


def _merge_stem_results(
  stem_segments: Dict[str, List[ChordSegment]],
  activities: Optional[Dict[str, StemActivity]] = None,
) -> List[ChordSegment]:
  activities = activities or {}

  # primary = mix 以外の分離 stem。これらが無い場合は mix を主役として投票する。
  primary_stems = [s for s in stem_segments if s != "mix"]
  has_primary = len(primary_stems) > 0
  voting_stems = primary_stems if has_primary else list(stem_segments)

  def _chord_at(stem: str, t: float) -> str:
    return next((s.label for s in stem_segments[stem] if s.start <= t < s.end), "N")

  # 全stemの境界時刻を収集してタイムラインを構築（mix の境界も含める）
  boundaries = sorted({
    t
    for segs in stem_segments.values()
    for seg in segs
    for t in (seg.start, seg.end)
  })

  merged = []
  for t0, t1 in zip(boundaries[:-1], boundaries[1:]):
    votes: Dict[str, float] = {}
    for stem in voting_stems:
      # 無音ゲート: この区間で stem が鳴っていなければ投票させない。
      # bass 無音区間では bass の "N" 票が捨てられ、other/vocals が和声を決める。
      activity = activities.get(stem)
      if activity is not None and not _stem_active(activity, t0, t1):
        continue
      chord = _chord_at(stem, t0)
      if stem == "bass":
        chord = root_only(chord)  # bass はルート音のみで投票
      votes[chord] = votes.get(chord, 0) + STEM_WEIGHTS.get(stem, 0)
    # 全 stem が無音 → 票が無い区間は "N"（No Chord）
    best = max(votes, key=votes.get) if votes else "N"

    # mix fallback: primary 投票が N の区間のみ、鳴っている mix のコードで穴埋めする。
    # 既存の非 N 検出は上書きしないため回帰リスクは無い。
    if has_primary and best == "N":
      mix_act = activities.get("mix")
      if mix_act is None or _stem_active(mix_act, t0, t1):
        mix_chord = _chord_at("mix", t0)
        if mix_chord != "N":
          best = mix_chord

    merged.append(ChordSegment(start=t0, end=t1, label=best, label_raw=best))

  return merged


def root_only(label: str) -> str:
  """コードラベルからルート音のみ抽出。例: Am → A, G7 → G, N → N"""
  if label in ('N', 'X', ''):
    return label
  m = _ROOT_RE.match(label)
  return m.group(1) if m else label


_ROOT_RE = re.compile(r'^([A-G][#b]?)')


def normalize_chord(label_raw: str) -> str:
  """madmom の root:quality 形式と従来形式の両方に対応して正規化する。

  例: C:maj → C, A:min → Am, G:maj7 → G, Bm7 → Bm, N → N
  """
  if label_raw in ('N', 'X', ''):
    return label_raw

  m = _ROOT_RE.match(label_raw)
  if not m:
    return label_raw
  root = m.group(1)
  rest = label_raw[len(root):]

  # madmom の "root:quality" 形式
  if rest.startswith(':'):
    quality = rest[1:]
    if quality.startswith('min'):
      return root + 'm'
    return root

  # 従来形式: Bm7, Cmaj7, Am7 など
  if rest.startswith('m') and not rest.startswith('maj'):
    return root + 'm'
  return root


def snap_to_beats(
  segments: List[ChordSegment],
  beats: List[float],
  downbeats: List[float],
) -> List[ChordSegment]:
  if not segments:
    return []

  segments = segments.copy()  # 安全のためコピー

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
      confidence=getattr(seg, "confidence", None),
    ))

    prev_end = new_end

  # 順序重要：まず同一ラベル結合 → 次に短区間除去
  snapped = _merge_consecutive_same_label(snapped)
  snapped = _merge_short_segments(snapped)

  return snapped


def _collect_candidates(
  t: float,
  beats: List[float],
  downbeats: List[float],
  window: float = SNAP_WINDOW,
) -> List[float]:
  candidates = [b for b in beats if abs(b - t) <= window]
  candidates += [d for d in downbeats if abs(d - t) <= window]
  return list(set(candidates))

def _is_near_downbeat(c, downbeats):
  return any(abs(c - d) < DOWNBEAT_TOL for d in downbeats)

def _is_close(a: float, b: float, tol: float = FLOAT_TOL) -> bool:
  return abs(a - b) < tol


def _choose_boundary(
  t: float,
  beats: List[float],
  downbeats: List[float],
  prev_boundary: Optional[float] = None,
  next_boundary: Optional[float] = None,
  side: str = 'start',
) -> float:
  # --- Phase 1: downbeat強制スナップ ---
  # t が DOWNBEAT_TOL 以内の downbeat に近ければ、ハード制約を満たす最近傍 downbeat を採用する。
  # 音楽理論的にコードチェンジは小節頭（downbeat）で起きることが最多のため、
  # 推定器の時間誤差がボーナス値を上回っても downbeat を優先する。
  nearby_downbeats = sorted(
    [d for d in downbeats if abs(d - t) <= DOWNBEAT_TOL],
    key=lambda d: abs(d - t),
  )
  for c in nearby_downbeats:
    if side == 'start':
      if prev_boundary is not None and c < prev_boundary - FLOAT_TOL:
        continue
      if next_boundary is not None and (next_boundary - c) < MIN_CHORD_DURATION:
        continue
    else:  # side == 'end'
      if prev_boundary is not None and (c - prev_boundary) < MIN_CHORD_DURATION:
        continue
      if next_boundary is not None and c > next_boundary + FLOAT_TOL:
        continue
    return c

  # --- Phase 2: スコアベースフォールバック ---
  candidates = _collect_candidates(t, beats, downbeats)

  if not candidates:
    return t

  best = None
  best_score = float('inf')

  for c in candidates:
    score = abs(c - t)

    # downbeat優遇（float安全比較）
    if any(_is_close(c, d) for d in downbeats):
      score -= DOWNBEAT_BONUS

    # 順序制約（強め）
    if side == 'start' and prev_boundary is not None:
      if c < prev_boundary:
        score += ORDER_PENALTY * 2

    if side == 'end' and next_boundary is not None:
      if c > next_boundary:
        score += ORDER_PENALTY * 2

      # 境界近すぎペナルティ
      if abs(c - next_boundary) < 0.05:
        score += 0.2

    # 最小長チェック（統一）
    duration = None
    if side == 'start' and next_boundary is not None:
      duration = next_boundary - c
    elif side == 'end' and prev_boundary is not None:
      duration = c - prev_boundary

    if duration is not None and duration < MIN_CHORD_DURATION:
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

  segments = segments.copy()

  changed = True
  while changed:
    changed = False
    result = []
    i = 0

    while i < len(segments):
      seg = segments[i]

      if (seg.end - seg.start) < min_duration:
        if i == 0 and len(segments) > 1:
          # 次に統合
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
          # 前に統合
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
