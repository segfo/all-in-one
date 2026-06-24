# 03. ROCm / AMD GPU 対応

> 実装: [`src/allin1/rocm_patch.py`](../src/allin1/rocm_patch.py),
> [`rocm_patch.cpu.py`](../src/allin1/rocm_patch.cpu.py),
> [`models/dinat.py`](../src/allin1/models/dinat.py),
> [`postprocessing/helpers.py`](../src/allin1/postprocessing/helpers.py)。
> 変更概要は [`CHANGELOG.md`](../CHANGELOG.md) も参照。

このフォークの動機のひとつが「**AMD GPU（ROCm / HIP）で動かす**」こと。
upstream は CUDA 前提で、ROCm 上では複数の数値カーネルバグにより NaN やハングが発生する。
本対応はそれらを既知バグごとに回避パッチとして実装している。

> 📌 ROCm の診断手法やバグの原理そのものは、**プロジェクト横断で使える独立した汎用ナレッジリポジトリ
> 「ROCm対応記録」**（本リポジトリの構成要素ではない別リポジトリ。`CUDAtoROCmPorting.md`,
> `BUG_CATALOG.md` 等）にまとめている。本ドキュメントは all-in-one 固有の適用内容のみを扱い、
> 原理の詳細はそちらに委譲する（重複を避けるため）。

---

## 1. 適用方針

すべてのパッチは ROCm 環境でのみ作用する。非 ROCm（CUDA / CPU / MPS）では no-op。

```python
def is_rocm() -> bool:
    return getattr(torch.version, 'hip', None) is not None
```

エントリポイント [`apply_rocm_patches()`](../src/allin1/rocm_patch.py) は、解析の冒頭
（[`analyze.py`](../src/allin1/analyze.py)）と音源分離の子プロセス
（[`demucs_runner.py`](../src/allin1/demucs_runner.py)）の両方で呼ばれる。
`is_rocm()` が False なら即 return するため、他環境の挙動は変わらない。

---

## 2. `apply_rocm_patches()` が適用するパッチ

| Bug | 症状 | 回避策 | 実装 |
|---|---|---|---|
| #4 | Flash / Mem-Efficient SDP が NaN を生成 | 両 SDP backend を無効化 | `enable_flash_sdp(False)` / `enable_mem_efficient_sdp(False)` |
| #1 | LayerNorm の大規模 reduction が NaN | GPU 上で fp32 アップキャストして計算 | `_patch_layer_norm` |
| #2 | GroupNorm の大規模 reduction が NaN | GPU 上で fp32 アップキャスト | `_patch_group_norm` / `_group_norm_gpu` |
| #3 | demucs sin_embedding の整数テンソル累乗が NaN | `exp(adim·log(max_period)/…)` の GPU ネイティブ実装に置換 | `_patch_sin_embedding` |
| #5 | fp16 Conv1d カーネルが NaN | GPU 上で fp32 アップキャスト | `_patch_conv1d` |
| #15 | demucs STFT / iSTFT が fp16 で精度問題 | 当該演算を CPU フォールバック | `_patch_stft_iSTFT` |
| #16 | demucs htdemucs の mean/std reduction が NaN | 当該演算を CPU フォールバック | `_patch_htdemucs_normalization` |

いずれも「GPU の特定カーネルだけが壊れている」ため、計算精度を保ちつつ
**問題のカーネルだけを fp32 化または CPU 退避**して回避する設計になっている。

> 💡 **なぜこれらのバグが出るのか（前提）**: [`analyze.py`](../src/allin1/analyze.py) は GPU 推論を
> `torch.amp.autocast` による **fp16 混合精度**で実行する（速度・メモリのため。CPU では無効化）。
> 上記バグ群は、この **fp16 経路で ROCm の壊れたカーネルが踏まれる**ことで顕在化する。
> つまり autocast(fp16) と本パッチ群は**セットで意味を持つ**——どちらか一方だけを消すと、
> 速度が落ちる（autocast 削除）か NaN が出る（パッチ削除）。`enabled=(device_type != 'cpu')` により
> CPU 実行では fp16 もパッチも作用しない。

---

## 3. その他の ROCm 固有対応

### NATTEN（Neighborhood Attention）の置換

upstream はモデルの近傍アテンションに NATTEN を使うが、ROCm では動作しない。
[`models/dinat.py`](../src/allin1/models/dinat.py) は、チェックポイント互換の
**純 PyTorch 実装** [`na1d_rocm` / `na2d_rocm`](../src/allin1/rocm_patch.py) に置き換えている
（旧 `natten1dqkrpb` / `natten2dqkrpb` のレイアウト互換）。
バックエンド選択が効かない問題のため、定数 `NATTEN_BACKEND_ROCM = 'flex-fna'` を明示指定する。

### `local_maxima` の GPU ネイティブ化（Bug #10）

[`postprocessing/helpers.py`](../src/allin1/postprocessing/helpers.py) の `local_maxima()` は、
`torch.eq` による float 厳密等値比較と boolean mask indexing（scattered write）が
ROCm 上で不正確になる問題（Bug #10）を持っていた。
`>= max_vals` 比較と `torch.where` に置き換えることで、**CPU 転送なしで GPU 上で完結**させた。
CUDA / ROCm / MPS / CPU すべてで動作する。

### GPU リソースの明示解放（Bug #12）

ROCm では、GPU テンソルが残ったままプロセス終了シーケンスに入ると HIP ランタイムが
デッドロックしてハングする。[`run_inference`](../src/allin1/helpers.py) は `finally` で
推論ごとに `spec` / `logits` を解放し、`synchronize()` + `empty_cache()` する
（[`release_gpu_resources`](../src/allin1/rocm_patch.py) も同目的）。

### torchaudio のロードパッチ

[`patch_torchaudio_load`](../src/allin1/rocm_patch.py) を `apply_rocm_patches()` と併せて適用する。

---

## 4. MPS（Apple Silicon）フォールバック

ROCm 対応と同じく、デフォルトデバイス選択を拡張している（[`CHANGELOG.md`](../CHANGELOG.md)）。

- [`analyze.py`](../src/allin1/analyze.py) / [`cli.py`](../src/allin1/cli.py): CUDA 未検出時に
  MPS が利用可能なら `'mps'` を選択する
- [`models/loaders.py`](../src/allin1/models/loaders.py): CUDA チェックを `device_count()` から
  `is_available()` に統一

---

## 関連ドキュメント

- [00_overview.md](00_overview.md) — リポジトリ全体像
- 外部: 汎用ナレッジリポジトリ「ROCm対応記録」（`CUDAtoROCmPorting.md`, `BUG_CATALOG.md`）— 本リポジトリ外
