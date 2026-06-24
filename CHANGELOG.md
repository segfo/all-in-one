# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- `chord_detection.py`: コード認識器を madmom の DeepChroma（`DeepChromaProcessor` +
  `DeepChromaChordRecognitionProcessor`）から **CNN ベース**（`CNNChordFeatureProcessor` +
  `CRFChordRecognitionProcessor`）へ差し替え。正解曲（`01 - 夜に駆ける`）で退行なし
  （種別精度 92%→96%、ルート 83%・N 2%・ピッチずれ無しは同等）、synth/EDM 系の難曲で
  N（No Chord）を大幅削減（認識器単体で 46%→9% 等）。stem 投票・無音ゲート・mix fallback の
  架構は維持。DeepChroma 専用だった numpy 2.x 対策パッチ（`_patch_madmom_crf`）を撤去。

### Added

- `chord_detection.py`: stem 無音ゲートを追加。各区間で stem のフレーム RMS が自ピークの
  相対閾値（`STEM_SILENCE_REL`）未満なら投票から除外する。これにより**ベースが鳴っていない
  区間（イントロ等）で bass の `N` 票が `other` の実コードを潰す問題**を解消し、bass 無音時は
  `other`/`vocals` がコードを決めるようになった。
- `chord_detection.py`: mix フォールバックを追加。`mix`（オリジナル音源）を常に併走させ、
  **primary（bass/other/vocals）投票が `N` の区間に限り** mix のコードで穴埋めする
  （既存の非 `N` 検出は上書きしないため回帰リスク無し）。分離 stem 単体では `N` でも
  全和声が混ざった mix なら拾える区間があるため。synth/EDM 系 2 曲で N 率 64%→36% / 59%→28%。
  madmom 認識を mix にも 1 回追加実行するコストがかかる。

### Changed

- `postprocessing/helpers.py`: `local_maxima()` を CPU 転送なしの GPU ネイティブ実装に変更。
  `torch.eq` → `>= max_vals` に置き換えて float 厳密等値比較を排除し、
  boolean mask indexing（scattered write）を `torch.where` に置き換えることで
  ROCm Bug #6（GPU 上の比較演算不正確）を CPU 転送なしで回避。CUDA/ROCm/MPS/CPU すべてで動作。
- `analyze.py` / `cli.py`: デフォルトデバイス選択に MPS (Apple Silicon) フォールバックを追加。
  CUDA 未検出時に MPS が利用可能な場合は `'mps'` を選択するようになった。
- `models/loaders.py`: デバイス自動選択の CUDA チェックを `device_count()` から `is_available()` に統一。
  また CUDA 非検出時に MPS (Apple Silicon) を検出してから CPU にフォールバックするよう拡張。
- `analyze.py`: `torch.amp.autocast` の device_type が `'cuda'` にハードコードされていた問題を修正。
  `device` 変数から動的に device_type を導出することで ROCm / MPS でも正しく動作するようになった。
- `rocm_patch.py`: `_rocm_create_sin_embedding` のデフォルト引数 `device='cpu'` を `None` に変更。
  引数省略時は CUDA → MPS → CPU の順でランタイム検出するようになった。
- `models/allinone.py`: `reset_parameters` 内の `torch.tensor(...)` を `math.log` スカラーに置き換え。
  デバイス未指定の `torch.tensor` は常に CPU テンソルを生成するため、モデルを GPU に移動後に
  `fill_()` へ渡すとデバイス不一致になる (ROCm Bug #17)。Python スカラーに置き換えることで回避。

## [1.1.0] - 2023-10-10

### Added

- Training code and instructions.

[unreleased]: https://github.com/mir-aidj/all-in-one/compare/v1.1.0...HEAD
[1.1.0]: https://github.com/olivierlacan/keep-a-changelog/compare/v1.0.3...v1.1.0
