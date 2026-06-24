import numpy as np
import json
import librosa
import torch

from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from glob import glob
from typing import Dict, List, Optional, Union
from .utils import mkpath, compact_json_number_array
from .typings import AllInOneOutput, AnalysisResult, PathLike, Segment
from .postprocessing import (
  postprocess_metrical_structure,
  postprocess_functional_structure,
  estimate_tempo_from_beats,
  estimate_bpm_from_audio,
  estimate_bpm_per_segment_from_audio,
  estimate_bpm_from_downbeats,
)


def _fuse_bpm(
  bpm_from_beats: Optional[int],
  audio_result: dict,
  beats_count: int = 0,
  tolerance: int = 5,
) -> Optional[int]:
  """ビートベースと tempogram ベースの BPM 推定を統合して最終 BPM を決定する。

  - ビート数が少ないほど beat-based の信頼性が低いため、tolerance をビート数に応じて動的に調整する（改善 9-1）。
  - 両者が直接一致、または半速/倍速の関係で合意している場合は beat-based を採用する（改善 9-2）。
  - 合意しない場合は tempogram を採用する。
  """
  audio_bpm = float(audio_result.get("tempo") or 0.0)

  if bpm_from_beats is None:
    return int(round(audio_bpm)) if audio_bpm else None
  if not audio_bpm:
    return bpm_from_beats

  # ビート数に応じて tolerance を動的調整（改善 9-1）
  dynamic_tolerance = max(3, 10 - beats_count // 10)
  effective_tolerance = min(tolerance, dynamic_tolerance)

  # 直接一致 or 半速/倍速の関係で合意していれば beat-based を採用（改善 9-2）
  for factor in (1.0, 0.5, 2.0):
    if abs(bpm_from_beats - audio_bpm * factor) <= effective_tolerance:
      return bpm_from_beats

  # 乖離している → tempogram を採用
  return int(round(audio_bpm))


def _select_representative_bpm(
  segments: List[Segment],
  current_bpm: Optional[int],
  tempo_candidates: List[float],
  tolerance: int = 5,
) -> Optional[int]:
  """セクション別 BPM のアンサンブルで代表 BPM を検証・補正する。

  1. セクション BPM を継続時間で重み付け投票
  2. ±tolerance BPM 以内 or 半速/倍速を同クラスタに集約
  3. 最大重みクラスタ → dominant_bpm
  4. tempo_candidates と harmonic-aware 照合 → best_candidate（候補の元スケールを維持）
  5. 最終選出:
     - |current - best| ≤ 2 → current 維持（小差は beat-based を信頼）
     - オクターブ関係 かつ current スケール支持セクションあり → current 維持
     - それ以外 → best_candidate 採用
  """
  # Step 1: 継続時間重み付き投票
  weighted_votes: Dict[int, float] = defaultdict(float)
  for seg in segments:
    if seg.bpm is not None and seg.bpm > 0:
      weighted_votes[seg.bpm] += seg.end - seg.start

  if not weighted_votes:
    return current_bpm

  # Step 2: Harmonic-aware クラスタリング（重みの大きい BPM からシードとして処理）
  clustered: Dict[int, float] = {}
  for bpm_val in sorted(weighted_votes, key=weighted_votes.__getitem__, reverse=True):
    weight = weighted_votes[bpm_val]
    placed = False
    for c in list(clustered):
      if abs(c - bpm_val) <= tolerance:
        clustered[c] += weight
        placed = True
        break
      for factor in (0.5, 2.0):
        if abs(c - bpm_val * factor) <= tolerance:
          clustered[c] += weight
          placed = True
          break
      if placed:
        break
    if not placed:
      clustered[bpm_val] = weight

  # Step 3: 最大重みクラスタ → dominant_bpm
  dominant_bpm = max(clustered, key=clustered.__getitem__)

  # Step 4: tempo_candidates との harmonic-aware 照合（候補の元スケールを維持）
  best_candidate: Optional[int] = None
  best_dist = float('inf')
  for cand in tempo_candidates:
    for factor in (1.0, 0.5, 2.0):
      dist = abs(dominant_bpm - cand * factor)
      if dist < best_dist:
        best_dist = dist
        best_candidate = int(round(cand))

  # Step 5: 最終選出
  if current_bpm is None:
    return best_candidate
  if best_candidate is None:
    return current_bpm

  # ±2 BPM 以内（丸め誤差・小差）→ beat-based の current_bpm を維持
  if abs(current_bpm - best_candidate) <= 2:
    return current_bpm

  # オクターブ関係の場合、セクション内に current スケールのサポートがあれば維持
  # 例: ファンタスティック（current=109, best=216, instに110あり → 109維持）
  # 例: 戦場の華（current=97, best=194, 97近傍セクションなし → 194採用）
  for factor in (0.5, 2.0):
    if abs(current_bpm - best_candidate * factor) <= tolerance:
      if any(
        seg.bpm is not None and abs(seg.bpm - current_bpm) <= tolerance
        for seg in segments
      ):
        return current_bpm
      break  # サポートなし → best_candidate へ

  return best_candidate


def _correct_half_speed_beats(
  bpm: Optional[int],
  beats: List[float],
  downbeats: List[float],
  beat_positions: List[int],
  min_half_speed_ratio: float = 0.05,
) -> Optional[tuple]:
  """BPMに対してハーフスピードなビート間隔を検出し、中点補間で修正する。

  確定済みBPMから期待ビート間隔を算出し、各ビートギャップが
  「期待間隔の 1.7〜2.3 倍」に該当するかを個別に判定する。
  そのようなギャップが全体の min_half_speed_ratio 以上存在する場合に補正を実施。

  グローバル中央値による判定（全曲ハーフスピードの場合のみ有効）と異なり、
  曲の一部だけがハーフスピードの混在パターンにも対応する。

  補正が不要な場合は None を返す。
  """
  if bpm is None or len(beats) < 4:
    return None

  expected_interval = 60.0 / bpm
  intervals = np.diff(beats)

  # ハーフスピードギャップ：期待間隔の 1.7〜2.3 倍
  hs_low = expected_interval * 1.7
  hs_high = expected_interval * 2.3
  half_speed_mask = (intervals >= hs_low) & (intervals <= hs_high)
  half_speed_ratio = float(np.sum(half_speed_mask)) / len(intervals)

  if half_speed_ratio < min_half_speed_ratio:
    return None  # 補正不要

  # ハーフスピードギャップにのみ中点を補間
  new_beats = []
  for i in range(len(beats) - 1):
    new_beats.append(beats[i])
    if half_speed_mask[i]:
      new_beats.append((beats[i] + beats[i + 1]) / 2.0)
  new_beats.append(beats[-1])

  # ダウンビートも同様に処理（1小節 = beats_per_bar ビートとして期待間隔を計算）
  beats_per_bar = max(beat_positions) if beat_positions else 4
  expected_db_interval = expected_interval * beats_per_bar
  db_hs_low = expected_db_interval * 1.7
  db_hs_high = expected_db_interval * 2.3

  if len(downbeats) >= 2:
    db_intervals = np.diff(downbeats)
    db_half_speed_mask = (db_intervals >= db_hs_low) & (db_intervals <= db_hs_high)
    new_downbeats = []
    for i in range(len(downbeats) - 1):
      new_downbeats.append(downbeats[i])
      if db_half_speed_mask[i]:
        new_downbeats.append((downbeats[i] + downbeats[i + 1]) / 2.0)
    new_downbeats.append(downbeats[-1])
    new_downbeats.sort()
  else:
    new_downbeats = list(downbeats)

  # beat_positions を再計算（ダウンビートでカウンタをリセット）
  db_list = sorted(new_downbeats)
  db_idx = 0
  counter = 0
  new_beat_positions = []
  for beat in new_beats:
    while db_idx + 1 < len(db_list) and beat >= db_list[db_idx + 1] - 1e-6:
      db_idx += 1
    if abs(beat - db_list[db_idx]) < 1e-4:
      counter = 1
    else:
      counter += 1
    new_beat_positions.append(counter)

  return new_beats, new_downbeats, new_beat_positions


def _correct_with_downbeat_bpm(
  bpm: Optional[int],
  beats: List[float],
  downbeats: List[float],
  tolerance: int = 3,
) -> Optional[int]:
  """ダウンビート間隔BPMで最終BPMを精度補正する。

  _select_representative_bpm() の結果に対してのみ適用する「精密層」。
  小節単位BPM（実測値）が現在BPMの ±tolerance 内なら小節単位BPMを採用する。
  範囲外なら変更しない（既存アルゴリズムの結果を信頼する）。

  テンポグラム量子化誤差（±3BPM 程度）を補正するために使用する。
  """
  if bpm is None:
    return bpm
  downbeat_bpm = estimate_bpm_from_downbeats(beats, downbeats)
  if downbeat_bpm is None:
    return bpm
  if abs(bpm - downbeat_bpm) <= tolerance:
    return downbeat_bpm
  return bpm


def _relabel_end_segments_by_rms(
  segments: list,
  y: np.ndarray,
  sr: int,
  beats: List[float] = None,
  rms_threshold: float = 0.01,
  hop_size: float = 0.05,
  min_duration: float = 1.0,
) -> list:
  """
  'end' ラベルのセグメントをフレームごとの RMS で解析し、outro / end に分割する。

  - beats が 0 の場合：Outro セグメントは End とみなす（ビートがないため）
  - 全フレームが閾値以下        → 変更なし（end のまま）
  - 全フレームが閾値超          → セグメント全体を outro に変換
  - 途中から無音に転落          → 音あり部分を outro、無音部分を end に分割
  min_duration 未満の端数セグメントが出る場合は分割せず全体を outro にする。
  """
  # beats が 0 の場合、Outro セグメントは End とみなす
  has_beats = beats is not None and len(beats) > 0

  hop_samples = int(hop_size * sr)
  result = []
  did_split = False
  for seg in segments:
    if seg.label != 'end':
      # beats がなく Outro セグメントの場合、End に変更
      if not has_beats and seg.label == 'outro':
        seg.label = 'end'
      result.append(seg)
      continue

    start_sample = int(seg.start * sr)
    end_sample = int(seg.end * sr)
    chunk = y[start_sample:end_sample]
    if len(chunk) == 0:
      result.append(seg)
      continue

    # フレームごとの RMS を計算
    n_frames = max(1, len(chunk) // hop_samples)
    frame_rms = np.array([
      np.sqrt(np.mean(chunk[i * hop_samples:(i + 1) * hop_samples] ** 2))
      for i in range(n_frames)
    ])

    above = np.where(frame_rms > rms_threshold)[0]
    if len(above) == 0:
      # 全フレームが無音 → end のまま
      result.append(seg)
      continue

    last_audio_frame = above[-1]
    split_time = seg.start + (last_audio_frame + 1) * hop_size

    outro_duration = split_time - seg.start
    end_duration = seg.end - split_time

    if outro_duration >= min_duration and end_duration >= min_duration:
      # 有効な分割点あり → outro + end の2セグメントに分割
      result.append(Segment(start=seg.start, end=split_time, label='outro', bpm=seg.bpm))
      result.append(Segment(start=split_time, end=seg.end, label='end', bpm=seg.bpm))
      did_split = True
    else:
      # 分割しても端数が短すぎる → 全体を outro に変換
      seg.label = 'outro'
      result.append(seg)

  # 分割が発生した場合のみ、連続する end セグメントを結合する
  if did_split:
    merged = []
    for seg in result:
      if merged and merged[-1].label == 'end' and seg.label == 'end':
        merged[-1] = Segment(start=merged[-1].start, end=seg.end, label='end', bpm=merged[-1].bpm)
      else:
        merged.append(seg)
    return merged

  return result


def run_inference(
  path: Path,
  spec_path: Path,
  model: torch.nn.Module,
  device: str,
  include_activations: bool,
  include_embeddings: bool,
) -> AnalysisResult:
  spec = None
  logits = None
  try:
    # 生音声をロードして tempogram ベース推定を実行する（NN に非依存）
    y, sr = librosa.load(str(path), sr=None, mono=True)
    audio_result = estimate_bpm_from_audio(y, sr)
    # y はセクション別 BPM 推定のために保持する（後で解放）

    spec = torch.from_numpy(np.load(spec_path)).unsqueeze(0).to(device)

    logits = model(spec)

    metrical_structure = postprocess_metrical_structure(logits, model.cfg)
    functional_structure = postprocess_functional_structure(logits, model.cfg)
    bpm_from_beats = estimate_tempo_from_beats(metrical_structure['beats'])
    bpm = _fuse_bpm(
      bpm_from_beats,
      audio_result,
      beats_count=len(metrical_structure['beats']),
    )

    # セクションごとの BPM を tempogram ロジックで推定
    segment_bpms = estimate_bpm_per_segment_from_audio(y, sr, functional_structure)
    for segment, seg_bpm in zip(functional_structure, segment_bpms):
      segment.bpm = seg_bpm

    # beats が 0 で Outro セグメントがある場合は End とみなす
    # 'end' ラベルかつ音声 RMS が有意なセグメントを 'outro' に変換
    functional_structure = _relabel_end_segments_by_rms(
      functional_structure, y, sr, metrical_structure['beats']
    )
    del y  # 以降 y は不要なので解放

    # セクション BPM のアンサンブルで代表 BPM を再選出
    # bpm_from_beats（全ビートのグローバル分布）を候補に追加して多テンポ曲対応を強化する
    extended_candidates = list(audio_result['tempo_candidates'])
    if bpm_from_beats is not None:
      # 既存候補と ±3 BPM 以内でなければ追加（重複防止）
      if not any(abs(bpm_from_beats - round(c)) <= 3 for c in extended_candidates):
        extended_candidates.append(float(bpm_from_beats))
    bpm = _select_representative_bpm(
      segments=functional_structure,
      current_bpm=bpm,
      tempo_candidates=extended_candidates,
    )

    # ダウンビート間隔BPM による精度補正（テンポグラム量子化誤差 ±3BPM 対策）
    bpm = _correct_with_downbeat_bpm(
      bpm=bpm,
      beats=metrical_structure['beats'],
      downbeats=metrical_structure['downbeats'],
      tolerance=3,
    )

    # 半速誤検出補正（BPM確定後に実施）
    corrected = _correct_half_speed_beats(
      bpm=bpm,
      beats=metrical_structure['beats'],
      downbeats=metrical_structure['downbeats'],
      beat_positions=metrical_structure['beat_positions'],
    )
    if corrected is not None:
      original_beats = metrical_structure['beats']
      original_downbeats = metrical_structure['downbeats']
      original_beat_positions = metrical_structure['beat_positions']
      metrical_structure['beats'], metrical_structure['downbeats'], metrical_structure['beat_positions'] = corrected
    else:
      original_beats = None
      original_downbeats = None
      original_beat_positions = None

    result = AnalysisResult(
      path=path,
      bpm=bpm,
      segments=functional_structure,
      tempo_candidates=audio_result['tempo_candidates'],
      original_beats=original_beats,
      original_downbeats=original_downbeats,
      original_beat_positions=original_beat_positions,
      **metrical_structure,
    )

    if include_activations:
      activations = compute_activations(logits)
      result.activations = activations

    if include_embeddings:
      result.embeddings = logits.embeddings[0].cpu().numpy()

    return result
  finally:
    # ROCm では GPU テンソルが残ったままプロセス終了シーケンスに入ると
    # HIP ランタイムがデッドロックする (CUDAtoROCmPorting.md Bug #12)。
    # 例外発生時も含め、推論ごとに確実に解放する。
    if spec is not None:
      del spec
    if logits is not None:
      del logits
    if torch.cuda.is_available():
      torch.cuda.synchronize()
      torch.cuda.empty_cache()


def compute_activations(logits: AllInOneOutput):
  activations_beat = torch.sigmoid(logits.logits_beat[0]).cpu().numpy()
  activations_downbeat = torch.sigmoid(logits.logits_downbeat[0]).cpu().numpy()
  activations_segment = torch.sigmoid(logits.logits_section[0]).cpu().numpy()
  activations_label = torch.softmax(logits.logits_function[0], dim=0).cpu().numpy()
  return {
    'beat': activations_beat,
    'downbeat': activations_downbeat,
    'segment': activations_segment,
    'label': activations_label,
  }


def expand_paths(paths: List[Path]):
  expanded_paths = set()
  for path in paths:
    if '*' in str(path) or '?' in str(path):
      matches = [Path(p) for p in glob(str(path))]
      if not matches:
        raise FileNotFoundError(f'Could not find any files matching {path}')
      expanded_paths.update(matches)
    else:
      expanded_paths.add(path)

  return sorted(expanded_paths)


def check_paths(paths: List[Path]):
  missing_files = []
  for path in paths:
    if not path.is_file():
      missing_files.append(str(path))
  if missing_files:
    raise FileNotFoundError(f'Could not find the following files: {missing_files}')


def rmdir_if_empty(path: Path):
  try:
    path.rmdir()
  except (FileNotFoundError, OSError):
    pass


def save_results(
  results: Union[AnalysisResult, List[AnalysisResult]],
  out_dir: PathLike,
  without_beats: bool = False,
):
  if not isinstance(results, list):
    results = [results]

  out_dir = mkpath(out_dir)
  out_dir.mkdir(parents=True, exist_ok=True)
  for result in results:
    out_path = out_dir / result.path.with_suffix('.json').name
    result_dict = asdict(result)
    result_dict['path'] = str(result_dict['path'])

    if without_beats:
      result_dict.pop('beats', None)
      result_dict.pop('downbeats', None)
      result_dict.pop('beat_positions', None)
      result_dict.pop('original_beats', None)
      result_dict.pop('original_downbeats', None)
      result_dict.pop('original_beat_positions', None)
    else:
      # 補正が行われなかった場合（None）はJSONに含めない
      for key in ('original_beats', 'original_downbeats', 'original_beat_positions'):
        if result_dict.get(key) is None:
          result_dict.pop(key, None)

    activations = result_dict.pop('activations')
    if activations is not None:
      np.savez(str(out_path.with_suffix('.activ.npz')), **activations)

    embeddings = result_dict.pop('embeddings')
    if embeddings is not None:
      np.save(str(out_path.with_suffix('.embed.npy')), embeddings)

    raw_chords = result_dict.pop('chords', None) or []
    if raw_chords:
      raw_chord_path = out_path.with_name(out_path.stem + '_raw_chord.json')
      raw_chord_path.write_text(json.dumps(raw_chords, indent=2))
    result_dict['chords'] = [
      {'start': c['start'], 'end': c['end'], 'label': c['label'], 'label_raw': c['label_raw'], 'confidence': c['confidence']}
      for c in raw_chords
    ] if raw_chords else None

    json_str = json.dumps(result_dict, indent=2)
    json_str = compact_json_number_array(json_str)
    out_path.with_suffix('.json').write_text(json_str)
