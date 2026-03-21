import numpy as np
from typing import List, Optional


def estimate_bpm_from_downbeats(
  beats: List[float],
  downbeats: List[float],
) -> Optional[int]:
  """ダウンビート間隔から小節単位BPMを推定する。

  テンポグラムの量子化に縛られない実測値を提供する。
  各小節区間内のビート数を数え、bar_duration から BPM を算出する。
  MAD外れ値除去後の中央値を返す。

  Args:
      beats: 全ビートのタイムスタンプ（秒）
      downbeats: ダウンビート（小節頭）のタイムスタンプ（秒）
  """
  if len(downbeats) < 2 or len(beats) < 2:
    return None

  downbeats_arr = np.array(sorted(downbeats))
  beats_arr = np.array(sorted(beats))

  bar_bpms = []
  for i in range(len(downbeats_arr) - 1):
    bar_start = downbeats_arr[i]
    bar_end = downbeats_arr[i + 1]
    bar_duration = bar_end - bar_start
    if bar_duration <= 0:
      continue
    beats_in_bar = int(np.sum((beats_arr >= bar_start) & (beats_arr < bar_end)))
    if beats_in_bar < 2:
      continue
    bar_bpms.append(beats_in_bar * 60.0 / bar_duration)

  if not bar_bpms:
    return None

  bar_bpms = np.array(bar_bpms)
  median = np.median(bar_bpms)
  mad = np.median(np.abs(bar_bpms - median))
  if mad > 0:
    bar_bpms = bar_bpms[np.abs(bar_bpms - median) <= 3 * mad]
  if len(bar_bpms) == 0:
    return None

  return int(round(float(np.median(bar_bpms))))


def estimate_tempo_from_beats(
  beats: List[float],
) -> Optional[int]:
  if len(beats) < 2:
    # The song has less than 2 beats. Perhaps it doesn't have much percussive elements.
    return None

  beats = np.array(beats)
  beat_interval = np.diff(beats)

  # MAD（中央絶対偏差）外れ値除去（改善 9-3）。
  # ライブ音源やルバートなどテンポが揺れる曲で精度を向上させる。
  median = np.median(beat_interval)
  mad = np.median(np.abs(beat_interval - median))
  if mad > 0:
    beat_interval = beat_interval[np.abs(beat_interval - median) <= 3 * mad]
  if len(beat_interval) == 0:
    return None

  bpm = 60. / beat_interval
  bpm = bpm.round().astype(int)
  bincount = np.bincount(bpm)
  bpm_range = np.arange(len(bincount))
  bpm_strength = bincount / bincount.sum()
  bpm_cand = np.stack([bpm_range, bpm_strength], axis=-1)
  bpm_cand = bpm_cand[np.argsort(bpm_strength)[::-1]]
  bpm_cand = bpm_cand[bpm_cand[:, 1] > 0]

  bpm_est = int(bpm_cand[0, 0])

  return bpm_est


def estimate_bpm_median_from_beats(
  beats: List[float],
) -> Optional[float]:
  """整数 bin を使わずにビート間隔の中央値から連続値 BPM を算出する。

  estimate_tempo_from_beats の bincount mode では整数丸め量子化誤差（±1〜2 BPM）が
  生じる場合がある。本関数はフロート中央値をそのまま返すため、
  _correct_with_downbeat_bpm と同様の ±tolerance 比較で精密補正に利用できる。

  Args:
      beats: 全ビートのタイムスタンプ（秒、昇順不問）
  """
  if len(beats) < 2:
    return None

  intervals = np.diff(np.array(sorted(beats)))
  median = np.median(intervals)
  mad = np.median(np.abs(intervals - median))
  if mad > 0:
    intervals = intervals[np.abs(intervals - median) <= 3 * mad]
  if len(intervals) == 0:
    return None

  return 60.0 / float(np.median(intervals))
