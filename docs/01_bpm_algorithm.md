# 01. BPM アルゴリズム — 設計と思想

> 実装: [`src/allin1/postprocessing/tempo_estimation.py`](../src/allin1/postprocessing/tempo_estimation.py),
> [`tempo.py`](../src/allin1/postprocessing/tempo.py), [`helpers.py`](../src/allin1/helpers.py),
> [`typings.py`](../src/allin1/typings.py)。検証スクリプト: [`compare_bpm.py`](../compare_bpm.py)。

upstream の all-in-one は BPM をビート間隔から単純に推定するが、本フォークでは
**性質の異なる 4 つのソースを融合**して、より人手ラベルに近い BPM を得る多層アルゴリズムに拡張した。

> 本書は「なぜこの構成か」の設計思想。**関数レベルの仕様**（各関数の入出力・全パラメータ・
> 出力フィールド・デバッグ手順）は [05_bpm_reference.md](05_bpm_reference.md)、
> **改良の実験経緯**（試行・差し戻しの理由）は [06_bpm_phase3_report.md](06_bpm_phase3_report.md) を参照。

---

## 1. 設計思想 — なぜ単一ソースでは足りないか

単一の推定法にはそれぞれ固有の弱点がある。

- **テンポグラム**（スペクトル分析）: 量子化誤差で ±数 BPM ずれる
- **全ビート中央値**: テンポ揺れのある曲で実用 BPM から乖離する
- **どちらも**: 多テンポ曲（曲中でテンポが変わる）では代表値を誤りやすい

そこで、弱点の異なる 4 ソースを組み合わせて互いを補正する。

| ソース | 性質 | 役割 |
|---|---|---|
| **beat-based** | ビート間隔の高精度計測 | ビートレベルの精度 |
| **tempogram** | スペクトル分析による候補（量子化誤差あり） | 候補の母集団 |
| **section（セクション別）** | 多点測定 | dominant tempo を空間的にカバー（多テンポ対策） |
| **downbeat（小節実測）** | 小節単位の実測（量子化に縛られない） | 精密補正層 |

---

## 2. 処理パイプライン

実行は [`run_inference`](../src/allin1/helpers.py) が統括する。

```
音声ロード
  ↓
① tempogram 推定        estimate_bpm_from_audio()        → tempo + tempo_candidates（NN非依存）
  ↓
② beat-based 推定       estimate_tempo_from_beats()      → bpm_from_beats
  ↓
③ 融合                  _fuse_bpm()                      → 暫定 bpm
  ↓
④ セクション別 BPM       estimate_bpm_per_segment_from_audio() → 各 Segment.bpm
  ↓
⑤ 代表 BPM 再選出        _select_representative_bpm()     → セクションアンサンブルで補正
  ↓
⑥ ダウンビート精密補正    _correct_with_downbeat_bpm()     → 量子化誤差 ±3BPM を補正
  ↓
⑦ 半速誤検出補正         _correct_half_speed_beats()      → beats/downbeats を補正（原値も保持）
  ↓
最終 BPM
```

### ③ 融合（[`_fuse_bpm`](../src/allin1/helpers.py)）

beat-based と tempogram を統合する。

- ビート数が少ないほど beat-based の信頼性が低いため、**tolerance をビート数に応じて動的調整**
- 両者が直接一致、または **半速/倍速の関係（×0.5, ×1.0, ×2.0）で合意**していれば beat-based を採用
- 乖離していれば tempogram を採用

### ⑤ 代表 BPM 再選出（[`_select_representative_bpm`](../src/allin1/helpers.py)）

セクション別 BPM のアンサンブルで代表値を検証・補正する。
多テンポ曲対策として、**全ビート分布から得た `bpm_from_beats` も候補に追加**してから選出する
（既存候補と ±3 BPM 以内でなければ追加）。
threshold=2 + オクターブ保護 + セクションサポートチェックで、安易な置換を防ぐ。

### ⑥ ダウンビート精密補正（[`_correct_with_downbeat_bpm`](../src/allin1/helpers.py) / [`estimate_bpm_from_downbeats`](../src/allin1/postprocessing/tempo.py)）

`_select_representative_bpm()` の結果に対する**精密層**。
各小節区間内のビート数を数え、`BPM = beats_in_bar × 60 / bar_duration` を小節ごとに実測。
MAD 外れ値除去後の中央値が、現在の BPM から ±3 BPM 以内なら採用する。
テンポグラムの量子化誤差を、量子化に縛られない小節実測値で補正するのが狙い。

> 設計原則: **全ビート中央値は過信禁物**。テンポ揺れのある曲で乖離する。
> 小節単位の実測（`_correct_with_downbeat_bpm`）の方が代替として堅牢。

### ⑦ 半速誤検出補正

BPM 確定後、ビートが半速で検出されているケースを補正する。
補正が発生した場合は、補正前の値を `original_beats` / `original_downbeats` /
`original_beat_positions` として保持する（`--sonify-original-beats` で利用可）。

---

## 3. tempo_candidates の品質改善（独立軸）

代表 BPM の精度とは別に、出力される `tempo_candidates` の質も改善している
（[`_deduplicate_scored_candidates`](../src/allin1/postprocessing/tempo_estimation.py)、
`get_tempo_voting` の出力段で適用）。

- ±3 BPM 近傍の近似値を除去
- 倍音関係でスコア比 < 0.7 の弱い側を除去

> 設計原則: **`tempo_candidates` 品質は代表 BPM 精度とは独立の改善軸**。
> 重複除去は BPM 精度には寄与しないが、候補情報の質として価値がある。

---

## 4. 検証結果（全 18 サンプル）

人手ラベル BPM との照合（[`compare_bpm.py`](../compare_bpm.py)）。

| 指標 | 結果 |
|---|---|
| exact 一致 | 13 / 18（72.2%） |
| ±2 BPM 以内 | 16 / 18（88.9%） |
| NG | 2 / 18 |

開発はフェーズで進めた。

- **フェーズ 1**: セクション別アンサンブル（③④⑤）+ `--without-beats` → exact 7/15
- **フェーズ 2**: ダウンビート精密補正（⑥）を追加 → exact 11/15（±1 誤差ケースを 4 件修正）
- **フェーズ 3**: `tempo_candidates` 品質改善 + `bpm_from_beats` を候補追加 → exact 13/18

### NG ケースの根本原因

- **MAXIMUM THE HORMONE [F]**（diff=9）: セクション BPM が 136〜207 と極めて散漫な多テンポ曲。
  最長時間を占める 152 を dominant として正しく選出しているが、人手 143 はどのセクション BPM とも一致しない。**多テンポ曲の本質的困難**。
- **救世主-月詠み**（diff=3）: tempo_candidates が [208, 104, 201] で、真値 205 が峰値間に落ちる。
  ダウンビート BPM が 208 から ±3 を超えたため補正されず。**テンポグラム量子化精度の限界**。

---

## 5. 関連 CLI オプション

| オプション | 説明 |
|---|---|
| `--without-beats` | JSON から `beats` / `downbeats` / `beat_positions`（および原値）を除外して BPM 中心の出力にする |
| `--sonify-original-beats` | 半速補正前の原ビートで sonify する |

定義は [`cli.py`](../src/allin1/cli.py)、適用は [`save_results`](../src/allin1/helpers.py)。

---

## 関連ドキュメント

- [05_bpm_reference.md](05_bpm_reference.md) — BPM 関数リファレンス（確定仕様・全パラメータ・デバッグ）
- [06_bpm_phase3_report.md](06_bpm_phase3_report.md) — フェーズ3改良の実験記録（試行・差し戻しの経緯）
- [00_overview.md](00_overview.md) — リポジトリ全体像
- [02_chord_detection.md](02_chord_detection.md) — コード検出
