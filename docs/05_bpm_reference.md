# 05. BPM 推定・決定ロジック — 関数リファレンス

> 設計思想・なぜこの構成かは [01_bpm_algorithm.md](01_bpm_algorithm.md)。**本書は関数レベルの仕様**
> （各関数の入出力・全パラメータ・出力フィールド・デバッグ手順）をまとめたリファレンス。
> 実装: [`postprocessing/tempo.py`](../src/allin1/postprocessing/tempo.py),
> [`postprocessing/tempo_estimation.py`](../src/allin1/postprocessing/tempo_estimation.py),
> [`helpers.py`](../src/allin1/helpers.py)。

---

## 1. 概要

allin1 の BPM 推定は **4 層のアルゴリズムを順番に組み合わせて**最終 BPM を決定する。

| 層 | アルゴリズム | 実装 | 役割 |
|---|---|---|---|
| 1 | ビートベース | [`tempo.py`](../src/allin1/postprocessing/tempo.py) | NN 検出ビートの間隔から BPM を計算（高精度） |
| 2 | tempogram ベース | [`tempo_estimation.py`](../src/allin1/postprocessing/tempo_estimation.py) | 生音声波形から独立推定（NN 非依存） |
| 3 | セクション別 BPM | [`tempo_estimation.py`](../src/allin1/postprocessing/tempo_estimation.py) | セクション単位で tempogram を再適用し多点計測 |
| 4 | ダウンビート補正 | [`tempo.py`](../src/allin1/postprocessing/tempo.py) | 小節単位の実測値で量子化誤差を精密補正 |

ニューラルネット（AllInOne + madmom DBN）のビート検出結果に依存する**層 1** は、ビートが正確に検出された場合に高精度だが、打楽器が少ない曲ではビートが取れず失敗する。
**層 2**（tempogram）はビート検出を経由せず音声波形から直接推定するため、層 1 が失敗した場合のフォールバックとして機能する。
**層 3**（セクション別 BPM）は複数のセクションで独立に推定した値をアンサンブルし、dominant tempo を統計的に検証する。
**層 4**（ダウンビート補正）は小節区間内のビート数を実測することでテンポグラムの量子化誤差（±3 BPM 程度）を除去する。

---

## 2. 各アルゴリズムの説明

### 2-A. ビートベース：`estimate_tempo_from_beats`

**ファイル：** [`src/allin1/postprocessing/tempo.py`](../src/allin1/postprocessing/tempo.py)

**入力：** `beats: List[float]` — madmom DBN が検出したビート時刻のリスト（単位：秒）

**処理手順：**

1. 隣接ビート間の時間間隔を計算する：`beat_interval = diff(beats)`
2. MAD（中央絶対偏差）外れ値除去：`|interval - median| > 3 * MAD` を除去（ルバート・ライブ音源対応）
3. BPM に換算する：`bpm = 60.0 / beat_interval`
4. 整数に丸めてヒストグラム（bincount）を作成する
5. 各 BPM 値の出現割合（信頼度）を計算し、最頻出 BPM を `int` で返す

**出力：** `int | None`（ビート数 < 2 の場合は `None`）

---

### 2-B. tempogram ベース：`estimate_bpm_from_audio`

**ファイル：** [`src/allin1/postprocessing/tempo_estimation.py`](../src/allin1/postprocessing/tempo_estimation.py)

**入力：** `y: np.ndarray`, `sr: int` — 生の音声波形とサンプリングレート

**処理手順：**

1. **onset エンベロープの計算：** `librosa.onset.onset_strength()` でビートの強度時系列を得る
2. **tempogram の計算：** onset エンベロープから自己相関ベースの tempogram を計算する
3. **区間投票：** 曲を `segment_duration`（デフォルト 30 秒）の区間に分割し、各区間の tempogram 平均のピーク上位 `top_k_peaks` 個に投票する
4. **クラスタリング：** ±1 BPM 以内を同一クラスタとしてスコアを集約し、量子化ノイズを除去する
5. **倍音補正：** 得票した BPM に対し半速（×0.5）・倍速（×2.0）にも 0.6 倍の重みで得票を加算する（40〜220 BPM 帯のみ）
6. **外れ値・重複除去（フェーズ3追加）：** `_deduplicate_scored_candidates` により ±3 BPM 近傍の近似値と倍音スコア比 < 70% の弱い候補を除去する
7. スコア順に上位 3 候補（クリーン）を返す

**出力：**
```python
{
  "tempo": float,                   # 最有力候補 BPM
  "tempo_candidates": List[float],  # 上位 3 候補（重複・弱倍音除去済み）
}
```

---

### 2-C. セクション別 BPM：`estimate_bpm_per_segment_from_audio` ✅ フェーズ1追加

**ファイル：** [`src/allin1/postprocessing/tempo_estimation.py`](../src/allin1/postprocessing/tempo_estimation.py)

**入力：** `y: np.ndarray`, `sr: int`, `segments: List[Segment]`

**処理手順：**

1. 各セクション（`start` 〜 `end`）の音声をスライス
2. セクション継続時間 < 4.0 秒なら `None`（推定不能）
3. セクション音声全体を 1 ブロックとして `estimate_bpm_from_audio` を適用
4. セクションごとの BPM を `Segment.bpm` に格納

**結果の使用：** `_select_representative_bpm()` が継続時間重み付き投票とクラスタリングで dominant BPM を選出し、現在 BPM を検証・補正する。

---

### 2-D. ダウンビート補正：`estimate_bpm_from_downbeats` ✅ フェーズ2追加

**ファイル：** [`src/allin1/postprocessing/tempo.py`](../src/allin1/postprocessing/tempo.py)

**入力：** `beats: List[float]`, `downbeats: List[float]`

**処理手順：**

1. 各小節区間 `[downbeats[i], downbeats[i+1])` 内のビート数を計数
2. `bar_bpm = beats_in_bar × 60 / bar_duration` で小節単位 BPM を実測
3. MAD 外れ値除去後の中央値を返す

**使用：** `_correct_with_downbeat_bpm(tolerance=3)` が最終 BPM との差が ±3 BPM 以内の場合のみ実測値に置き換える。tempogram の量子化誤差（±数 BPM）を除去する。

---

## 3. 長所と短所

| | ビートベース | tempogram ベース | セクション別 BPM | ダウンビート補正 |
|---|---|---|---|---|
| **入力** | NN 検出ビート時刻 | 生音声波形 | 生音声波形 + セクション | NN 検出ビート + ダウンビート |
| **長所** | ビート検出が正確なら最高精度。MAD 外れ値除去済み | NN に非依存。倍音補正あり | 多点計測で統計的に安定。多テンポ曲に対応 | テンポグラム量子化誤差を実測値で除去 |
| **短所** | ビート数 < 2 で `None`。半速/倍速補正は融合層で補う | 通常曲でビートベースより精度が劣る場合あり | 4 秒未満セクションはスキップ。計算コスト大 | NN beats が誤っていれば下流も誤る。大きな乖離は無視 |
| **適した曲** | 打楽器がはっきりした曲 | 打楽器が少ない曲 | 全曲（アンサンブル検証） | 全曲（精密補正） |

---

## 4. BPM の決定方法（全パイプライン）

### Stage 1: ビートベース × tempogram 融合（`_fuse_bpm`）

```
bpm_from_beats: int | None   ← estimate_tempo_from_beats() の結果
audio_result["tempo"]: float ← estimate_bpm_from_audio() の結果
beats_count: int             ← 検出ビート数（動的 tolerance の計算に使用）
```

1. `bpm_from_beats が None` → `audio_result["tempo"]` を採用
2. `audio_result["tempo"] が 0` → `bpm_from_beats` をそのまま採用
3. 動的 tolerance 計算：`effective_tolerance = min(5, max(3, 10 - beats_count // 10))`
4. 合意チェック（直接一致 or 半速/倍速の調和関係）：
   ```
   for factor in (1.0, 0.5, 2.0):
     |bpm_from_beats - audio_bpm * factor| <= effective_tolerance？
       Yes → bpm_from_beats を採用（合意 = beat-based を信頼）
   ```
5. いずれも不一致 → `audio_result["tempo"]` を採用（tempogram にフォールバック）

### Stage 2: セクション別 BPM アンサンブル（`_select_representative_bpm`）

1. 各 `segment.bpm` を継続時間で重み付け投票
2. ±5 BPM または半速/倍速関係を同クラスタに集約 → `dominant_bpm`
3. `extended_candidates`（`tempo_candidates` + `bpm_from_beats`）と harmonic-aware 照合 → `best_candidate`
4. 最終選出ルール：
   - `|current - best| ≤ 2` → current 維持（丸め誤差は beat-based を信頼）
   - オクターブ関係 かつ current スケールのセクションサポートあり → current 維持
   - それ以外 → `best_candidate` 採用

### Stage 3: ダウンビート精密補正（`_correct_with_downbeat_bpm`）

- 小節単位 BPM（実測値）を計算
- `|current_bpm - downbeat_bpm| ≤ 3` なら `downbeat_bpm` を採用（量子化誤差補正）
- 範囲外なら変更しない（既存アルゴリズムの結果を信頼）

---

## 5. 処理フロー（現在）

```
analyze.py（推論ループ）
  │
  └─ run_inference(path, spec_path, model, device, ...)
       │                                              [helpers.py]
       │
       ├─ librosa.load(path, sr=None, mono=True)
       │    └─ y: np.ndarray, sr: int
       │
       ├─ estimate_bpm_from_audio(y, sr)              [tempo_estimation.py]
       │    ├─ get_tempo_voting() → _deduplicate_scored_candidates()
       │    └─ audio_result: {"tempo": float, "tempo_candidates": List[float]}
       │
       ├─ np.load(spec_path) → spec  ← NN 用スペクトログラム
       │
       ├─ model(spec) → logits       ← AllInOne 推論
       │
       ├─ postprocess_metrical_structure(logits)      [metrical.py]
       │    └─ beats, downbeats, beat_positions
       │
       ├─ postprocess_functional_structure(logits)    [functional.py]
       │    └─ segments: List[Segment]
       │
       ├─ estimate_tempo_from_beats(beats)            [tempo.py]
       │    └─ bpm_from_beats: int | None
       │
       ├─ _fuse_bpm(bpm_from_beats, audio_result, beats_count)
       │    └─ bpm: int (Stage 1)
       │
       ├─ estimate_bpm_per_segment_from_audio(y, sr, segments)
       │    └─ segment.bpm[] を設定
       │
       ├─ del y  ← セクション別推定完了後に解放
       │
       ├─ extended_candidates = tempo_candidates + bpm_from_beats（重複除去付き）
       ├─ _select_representative_bpm(segments, bpm, extended_candidates)
       │    └─ bpm: int (Stage 2)
       │
       ├─ _correct_with_downbeat_bpm(bpm, beats, downbeats, tolerance=3)
       │    └─ bpm: int (Stage 3 = 最終 BPM 数値)
       │
       └─ AnalysisResult(bpm=bpm, tempo_candidates=tempo_candidates, ...)
```

> **BPM 数値の決定はここまで（Stage 1〜3）。** この後段に走る半速ビート誤検出補正（⑦
> `_correct_half_speed_beats`）と Outro リラベル（`_relabel_end_segments_by_rms`）は
> `beats` / `downbeats` / セクションラベルを調整するもので、BPM **数値**の決定対象外。
> これらは [01_bpm_algorithm.md](01_bpm_algorithm.md#⑦-半速誤検出補正) を参照。

---

## 6. 出力フィールド

`AnalysisResult`（[`typings.py`](../src/allin1/typings.py)）のフィールド：

| フィールド | 型 | 説明 |
|---|---|---|
| `bpm` | `int` | 最終決定 BPM（4層パイプライン通過後の値） |
| `tempo_candidates` | `List[float]` | tempogram ベースの上位候補 BPM（重複・弱倍音除去済み） |
| `segments[].bpm` | `int \| None` | セクションごとの推定 BPM（4秒未満セクションは `None`） |

**JSON 出力例：**

```json
{
  "bpm": 161,
  "tempo_candidates": [161.0, 156.0, 322.0],
  "beats": [0.42, 0.86, 1.30, "..."],
  "downbeats": [0.42, 1.77, 3.12, "..."],
  "segments": [
    {"start": 0.17, "end": 30.65, "label": "verse", "bpm": 161},
    "..."
  ]
}
```

---

## 7. パラメータ一覧

| パラメータ | デフォルト | 定義箇所 | 説明 |
|---|---|---|---|
| `tolerance` | `5` BPM | `_fuse_bpm()` | ビートベースと tempogram の差がこの値以内なら合意とみなす。実効値は `min(tolerance, max(3, 10 - beats_count // 10))` で動的決定 |
| `segment_duration` | `30.0` 秒 | `get_tempo_voting()` | tempogram を分割する区間長 |
| `hop_length` | `512` サンプル | `estimate_bpm_from_audio()` | onset エンベロープ・tempogram の時間解像度 |
| `top_k_peaks` | `2` | `get_tempo_voting()` | 各区間から投票する tempogram ピーク数 |
| `octave_weight` | `0.6` | `get_tempo_voting()` 内 | 半速/倍速への倍音補正の重み係数 |
| `near_threshold` | `3` BPM | `_deduplicate_scored_candidates()` | 同一候補とみなす近傍距離 |
| `octave_ratio_threshold` | `0.7` | `_deduplicate_scored_candidates()` | 倍音関係の候補を除去するスコア比の閾値 |
| `min_duration` | `4.0` 秒 | `estimate_bpm_per_segment_from_audio()` | セクション別BPM推定の最小セクション長 |
| `tolerance` (Stage 3) | `3` BPM | `_correct_with_downbeat_bpm()` | ダウンビート実測値を採用する許容差 |

---

## 8. デバッグ方法

`estimate_bpm_from_audio()` に `debug=True` を渡すと、中間スコアが標準出力に表示される。

```python
from allin1.postprocessing.tempo_estimation import estimate_bpm_from_audio
import librosa

y, sr = librosa.load("your_track.flac", sr=None, mono=True)
result = estimate_bpm_from_audio(y, sr, debug=True)
```

出力例：
```
Weighted votes: {133: 4.23, 160: 1.87, 80: 1.12, ...}
Clustered:      {133: 4.23, 160: 1.87, 80: 1.12, ...}
Final scores:   [(133, 6.72), (66, 2.54), (167, 1.87), ...]
Clean candidates: [133.0, 167.0]
```

| 出力 | 意味 |
|---|---|
| `Weighted votes` | 全区間の投票集計（クラスタリング前） |
| `Clustered` | ±1 クラスタリング後のスコア |
| `Final scores` | 倍音補正後の最終スコア（上位 10 件） |
| `Clean candidates` | 重複・弱倍音除去後の出力候補（フェーズ3追加） |

---

## 9. 改良・実装記録

実験の経緯（試行錯誤・差し戻しの詳細）は [06_bpm_phase3_report.md](06_bpm_phase3_report.md) を参照。

### ✅ 9-1. `tolerance` の動的調整（フェーズ1実装済み）

`_fuse_bpm()` 内で `dynamic_tolerance = max(3, 10 - beats_count // 10)` を計算し、`min(tolerance, dynamic_tolerance)` を実効値として使用。ビート数が少ないほど beat-based の信頼性を低く見積もる。

### ✅ 9-2. beat-based 側への半速/倍速補正（フェーズ1実装済み）

`_fuse_bpm()` の合意チェックにおいて `factor in (1.0, 0.5, 2.0)` でループし、「ビートベース=120 BPM、tempogram=60 BPM」のような半速/倍速の調和関係も合意とみなす。

### ✅ 9-3. ビート間隔の外れ値除去（フェーズ1実装済み）

`estimate_tempo_from_beats()` 内で MAD を計算し、`|interval - median| > 3 * MAD` を除去。ライブ音源・ルバートへの対応。

### ✅ 9-4. セクション単位の BPM 推定（フェーズ1実装済み）

`estimate_bpm_per_segment_from_audio()` を追加。セクションごとの BPM を `Segment.bpm` に格納し、`_select_representative_bpm()` でアンサンブル検証する。

### ✅ 9-5. `tempo_candidates` へのビートベース候補の統合（フェーズ3実装済み）

`_select_representative_bpm()` 呼び出し前に `bpm_from_beats` を `extended_candidates` に追加し、harmonic-aware 照合に組み込む（重複防止付き）。
`tempo_candidates` 出力フィールド自体は tempogram ベースのまま（後方互換）。

### ✅ 9-6. `tempo_candidates` の外れ値・重複除去（フェーズ3実装済み）

`_deduplicate_scored_candidates()` を `get_tempo_voting()` の出力段に追加。±3 BPM 近傍の近似値と倍音スコア比 < 70% の弱い候補を除去し、「意味的に異なる候補のみ」を返すようにした。

### ✅ 9-7. ダウンビート間隔による量子化誤差補正（フェーズ2実装済み）

`_correct_with_downbeat_bpm()` と `estimate_bpm_from_downbeats()` を追加。テンポグラム量子化誤差（±3 BPM）を小節単位の実測値で補正。フェーズ2で 4 曲の ±1 誤差を解消した（AWAKE, インフェリア, 阿修羅ちゃん, ファンタスティック）。

### 未解決: 多テンポ曲の代表 BPM 定義

MAXIMUM THE HORMONE [F] のようにセクション BPM が極めて散漫な曲では、「代表 BPM」の定義自体が曖昧。最長セクション支配（現行）か全ビート最頻値かは仕様レベルの問題。詳細は [06_bpm_phase3_report.md](06_bpm_phase3_report.md) を参照。

---

## 関連ドキュメント

| ドキュメント | 内容 |
|---|---|
| [01_bpm_algorithm.md](01_bpm_algorithm.md) | BPM アルゴリズムの設計と思想（なぜ 4 ソース融合か） |
| [06_bpm_phase3_report.md](06_bpm_phase3_report.md) | フェーズ3改良の実験記録（試行・差し戻しの経緯と設計知見） |
| [00_overview.md](00_overview.md) | リポジトリ全体像 |
