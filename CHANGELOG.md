# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
