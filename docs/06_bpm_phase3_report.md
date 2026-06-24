# 06. BPM 改良の実験記録 — フェーズ3

> **日付**: 2026-03-21
> 試行錯誤・差し戻しの経緯と、そこから得た設計知見を残す記録。確定仕様は
> [05_bpm_reference.md](05_bpm_reference.md)、設計思想は [01_bpm_algorithm.md](01_bpm_algorithm.md)。
> 18 曲のフル結果表は [01 §4](01_bpm_algorithm.md#4-検証結果全-18-サンプル) を参照（本書では再掲しない）。

---

## 概要

フェーズ2完了時点（15曲評価）の精度をベースラインとして、3つの改良を試みた。
うち 2 つを採用、1 つを差し戻した。

| 指標 | フェーズ2 (15曲) | フェーズ3 (18曲) |
|------|----------------|----------------|
| exact match | 11/15 (73.3%) | 13/18 (72.2%) |
| ±2BPM以内 | 13/15 (86.7%) | 16/18 (88.9%) |
| NG (差 ≥3) | 2曲 | 2曲 |

※ 新規3曲（ルカルカ★ナイトフィーバー、Nobodyknows_ococロオドル、千本桜）を追加。既存15曲の BPM 結果は変化なし。
ベースラインの NG 2曲は MAXIMUM THE HORMONE (+9)、救世主-月詠み (+3)。

---

## 実施した変更

### 実装済み: Step 1 — `tempo_candidates` の外れ値・重複除去

**ファイル**: [`src/allin1/postprocessing/tempo_estimation.py`](../src/allin1/postprocessing/tempo_estimation.py)

**変更内容**: `_deduplicate_scored_candidates` を追加し、`get_tempo_voting` の出力段で適用

- **±3 BPM 近傍の近似値**を後出し側（低スコア）として除去（例: 120 と 121 は同一候補）
- **倍音関係でスコア比 < 70%** の弱い側を除去（例: 60 BPM が 120 BPM のアーチファクトである場合）

**効果**: `tempo_candidates` が意味的に異なる候補のみを返すようになった。ユーザーが「候補BPMを参考にする」ユースケースでの品質向上。

---

### 実装済み: Step 3 — global beat BPM を候補プールに追加

**ファイル**: [`src/allin1/helpers.py`](../src/allin1/helpers.py)

**変更内容**: `_select_representative_bpm` を呼ぶ前に、`bpm_from_beats`（全ビートのグローバル bincount mode）を `tempo_candidates` に追加（±3 BPM 重複防止付き）

**狙い**: 多テンポ曲（MAXIMUM THE HORMONE）でセクション支配以外の候補を提供する

**結果**: MAXIMUM THE HORMONE は変化なし（+9 NG のまま）。`bpm_from_beats` が既存 `tempo_candidates` と近似値だったか、global beat mode が 152 を返しているため効果なし。

---

### 試行・差し戻し: Step 2 — ビート間隔中央値 BPM による補正

**内容**: 全ビート間隔の float 中央値から BPM を計算し、±3 以内なら上書き

**結果**: マジェスティック・サディスティックが 161 exact → 158 NG に**回帰**したため差し戻し

**原因分析**:
- `_correct_with_downbeat_bpm`（小節単位精密計測）がすでに 161 を確定していた
- 全ビートのグローバル中央値はテンポ揺れの影響を受け ~158 を返し、161 を上書きした
- `_correct_with_downbeat_bpm` と目的が重複するうえ精度が劣ることが判明

`estimate_bpm_median_from_beats` 関数自体は [`postprocessing/tempo.py`](../src/allin1/postprocessing/tempo.py) に保持（将来の局所的応用のため）。

---

## 残る NG ケースと根本原因

### MAXIMUM THE HORMONE [F]（差 +9）

- セクション BPM が 136 / 144 / 152 / 162 / 172 / 207 と極めて散漫
- 最長セクション（64.35 秒）が 152 BPM → アルゴリズムは正しく dominant=152 を選出
- 人手値 143 はいずれのセクション BPM とも一致しない
- **結論**: 多テンポ曲の根本的困難ケース。「代表 BPM」の定義自体が曖昧な曲。

### 救世主-月詠み（差 +3）

- tempogram・NN beats・ダウンビート計測が全て 208 BPM で合意
- 真値 205 は全推定器の量子化分解能の限界内に落ちている
- **結論**: 差 ±3 BPM はテンポグラム量子化精度の限界範囲。許容誤差内とも言える。

---

## フェーズ3で確立した設計知見

### 知見 1: global beat median BPM は過信禁物

全ビート間隔の float 中央値（`estimate_bpm_median_from_beats`）は、テンポ揺れのある曲では実用的 BPM から乖離する。小節単位計測（`_correct_with_downbeat_bpm`）のほうが堅牢であり、同一目的に対しては後者を優先すること。

### 知見 2: `tempo_candidates` の品質は独立した改善軸

代表 BPM の精度と候補リストの有意性は別問題。重複除去（Step 1）は BPM 精度の数値には寄与しないが、候補を「異なる解釈を示す有意な値のみ」に絞ることで情報品質として価値がある。

### 知見 3: 下流補正が正しい値を返している場合、上流の補正で上書きしない

`_correct_with_downbeat_bpm` → `estimate_bpm_median_from_beats` の順で適用した場合、後者が前者の精密な結果を破壊することがある。補正の優先順位は「より精密な計測手法（小節単位）> グローバル統計量（全ビート中央値）」とする。

---

## 現在のパイプライン構成（フェーズ3完了後）

```
[A] 生音声 → tempogram 区間投票
     └─ _deduplicate_scored_candidates  ← フェーズ3追加
     └─ audio_result { tempo, tempo_candidates (クリーン) }

[B] NN推論 → beats / downbeats 検出

[C] estimate_tempo_from_beats(beats) → bpm_from_beats

[D] _fuse_bpm(bpm_from_beats, audio_result) → bpm (暫定)

[E] セクション別 tempogram → segment.bpm[]

[F] extended_candidates = tempo_candidates + bpm_from_beats  ← フェーズ3追加
    _select_representative_bpm(segments, bpm, extended_candidates) → bpm

[G] _correct_with_downbeat_bpm(bpm, beats, downbeats, tolerance=3) → bpm (最終)
```

---

## 関連ドキュメント

| ドキュメント | 内容 |
|---|---|
| [01_bpm_algorithm.md](01_bpm_algorithm.md) | 設計と思想・18 曲フル結果表 |
| [05_bpm_reference.md](05_bpm_reference.md) | 関数リファレンス（確定仕様・全パラメータ） |
