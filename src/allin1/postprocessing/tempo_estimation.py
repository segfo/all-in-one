"""
tempogram ベースの BPM 推定ユーティリティ。

ニューラルネットのビート検出に依存しない、生音声波形からの
テンポ推定を担う。estimate_bpm_from_audio() が公開 API。
"""

from collections import defaultdict
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

if TYPE_CHECKING:
  from ..typings import Segment

import librosa
import numpy as np
from scipy.signal import find_peaks


def _deduplicate_scored_candidates(
    candidates_with_scores: List[Tuple[int, float]],
    near_threshold: int = 3,
    octave_ratio_threshold: float = 0.7,
) -> List[Tuple[int, float]]:
    """スコア降順リストから近傍重複・弱倍音候補を除去する。

    1. ±near_threshold BPM 以内の近似候補は後出し側（低スコア）を除去
    2. 2倍/0.5倍の倍音関係にある候補で、スコア比 < octave_ratio_threshold なら弱い側を除去
       （例: 60BPM の score が 120BPM の 70% 未満なら 60 は倍音アーチファクトとして除去）
    """
    result: List[Tuple[int, float]] = []
    for bpm, score in candidates_with_scores:
        if any(abs(bpm - kept_bpm) <= near_threshold for kept_bpm, _ in result):
            continue
        is_weak_octave = any(
            abs(bpm - round(kept_bpm * factor)) <= near_threshold
            and score / kept_score < octave_ratio_threshold
            for kept_bpm, kept_score in result
            for factor in (0.5, 2.0)
        )
        if not is_weak_octave:
            result.append((bpm, score))
    return result


def get_tempo_voting(
    y: np.ndarray,
    sr: int,
    segment_duration: float = 30.0,
    hop_length: int = 512,
    top_k_peaks: int = 2,
    debug: bool = False,
) -> List[float]:
    """tempogram の区間投票で BPM 候補を推定する。"""
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    tempogram = librosa.feature.tempogram(onset_envelope=onset_env, sr=sr, hop_length=hop_length)
    bpms = librosa.tempo_frequencies(n_bins=tempogram.shape[0], sr=sr, hop_length=hop_length)

    cols_per_segment = int(librosa.time_to_frames(segment_duration, sr=sr, hop_length=hop_length))
    if cols_per_segment < 1:
        cols_per_segment = tempogram.shape[1]

    weighted_votes: Dict[int, float] = defaultdict(float)

    for start in range(0, tempogram.shape[1], cols_per_segment):
        segment = tempogram[:, start:start + cols_per_segment]
        if segment.shape[1] < max(4, cols_per_segment // 4):
            continue
        local = np.mean(segment, axis=1)

        peaks, _ = find_peaks(local)
        if peaks.size == 0:
            continue

        peak_scores = local[peaks]
        order = np.argsort(peak_scores)[::-1]
        for idx in order[:top_k_peaks]:
            p = peaks[idx]
            bpm = float(bpms[p])
            score = float(peak_scores[idx])
            key = round(bpm)
            weighted_votes[key] += score

    # ±1 BPM 以内を同一クラスタに集約して量子化ノイズを除去する
    clustered: Dict[int, float] = defaultdict(float)
    for bpm_int, score in weighted_votes.items():
        placed = False
        for c in list(clustered.keys()):
            if abs(c - bpm_int) <= 1:
                clustered[c] += score
                placed = True
                break
        if not placed:
            clustered[bpm_int] += score

    # 半速/倍速への倍音補正（0.6 倍の重みで加算）
    final_scores: Dict[int, float] = defaultdict(float)
    for bpm_int, score in clustered.items():
        for factor in (0.5, 1.0, 2.0):
            norm = bpm_int * factor
            if 40 <= norm <= 220:
                final_scores[round(norm)] += score * (1.0 if factor == 1.0 else 0.6)

    sorted_candidates = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)

    # 近傍重複・弱倍音候補を除去してクリーンな候補リストを生成
    clean_candidates = _deduplicate_scored_candidates(sorted_candidates)
    candidates = [float(bpm) for bpm, _ in clean_candidates[:5]]

    if debug:
        print("Weighted votes:", dict(weighted_votes))
        print("Clustered:     ", dict(clustered))
        print("Final scores:  ", dict(sorted_candidates[:10]))
        print("Clean candidates:", candidates)

    return candidates[:3]


def tempo_with_correction(all_candidates: List[float]) -> List[float]:
    """候補 BPM に半速/倍速補正をかけて上位候補を返す。"""
    rounded = [int(round(x)) for x in all_candidates]

    scores: Dict[int, float] = defaultdict(float)
    for bpm in rounded:
        scores[bpm] += 1.0
        if bpm % 2 == 0:
            scores[bpm // 2] += 0.5
        scores[bpm * 2] += 0.5

    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [float(bpm) for bpm, _ in sorted_scores[:3]]


def estimate_tempo_candidates(
    y: np.ndarray,
    sr: int,
    hop_length: int = 512,
    segment_duration: float = 30.0,
    top_k_peaks: int = 2,
    debug: bool = False,
) -> List[float]:
    """テンポ候補推定の高レベル API（区間投票＋倍音補正）。"""
    return tempo_with_correction(
        get_tempo_voting(
            y=y,
            sr=sr,
            segment_duration=segment_duration,
            hop_length=hop_length,
            top_k_peaks=top_k_peaks,
            debug=debug,
        )
    )


def estimate_bpm_from_audio(
    y: np.ndarray,
    sr: int,
    hop_length: int = 512,
    segment_duration: float = 30.0,
    top_k_peaks: int = 2,
    debug: bool = False,
) -> Dict[str, object]:
    """メモリ上の波形配列から BPM と候補を推定する。

    Parameters
    ----------
    y : np.ndarray
        モノラル音声波形。
    sr : int
        サンプリングレート。
    debug : bool
        True にすると中間スコアを標準出力に表示する。

    Returns
    -------
    dict
        {"tempo": float, "tempo_candidates": List[float]}
    """
    candidates = estimate_tempo_candidates(
        y=y,
        sr=sr,
        hop_length=hop_length,
        segment_duration=segment_duration,
        top_k_peaks=top_k_peaks,
        debug=debug,
    )
    return {
        "tempo": float(candidates[0]) if candidates else 0.0,
        "tempo_candidates": [float(v) for v in candidates],
    }


def estimate_bpm_per_segment_from_audio(
    y: np.ndarray,
    sr: int,
    segments: List['Segment'],
    hop_length: int = 512,
    min_duration: float = 4.0,
) -> List[Optional[int]]:
    """各セクションの音声スライスに tempogram BPM 推定を適用する。

    Parameters
    ----------
    y : np.ndarray
        モノラル音声波形（全体）。
    sr : int
        サンプリングレート。
    segments : List[Segment]
        セクションリスト（start/end が秒単位）。
    hop_length : int
        tempogram 計算の hop サイズ。
    min_duration : float
        これ未満の短いセクションは推定をスキップして None を返す（秒）。

    Returns
    -------
    List[Optional[int]]
        各セクションの推定 BPM。短すぎる場合は None。
    """
    bpms = []
    for seg in segments:
        duration = seg.end - seg.start
        if duration < min_duration:
            bpms.append(None)
            continue

        start_sample = int(seg.start * sr)
        end_sample = int(seg.end * sr)
        y_seg = y[start_sample:end_sample]

        # セクション全体を 1 ブロックとして投票する
        result = estimate_bpm_from_audio(
            y=y_seg,
            sr=sr,
            hop_length=hop_length,
            segment_duration=duration,
        )
        tempo = result.get('tempo')
        bpms.append(int(round(tempo)) if tempo else None)

    return bpms
