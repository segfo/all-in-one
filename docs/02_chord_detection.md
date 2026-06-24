# 02. コード検出 — 設計と思想

> 本ドキュメントは現行実装 [`src/allin1/chord_detection.py`](../src/allin1/chord_detection.py) を正本として記述する。
> 旧 `04_downbeat_snap_design.md` / `05_chord_detection_implementation.md` を統合・更新したもの。

このリポジトリを特徴づける拡張機能のひとつ。楽曲構造解析（beats / downbeats / segments）の上に「コードレイヤー」を載せ、再生 UI 同期にも研究ログにも使える安定したコード列を出力する。

---

## 1. 狙い

madmom のコード認識器の生出力は、そのままでは次の理由で扱いにくい。

- 1 つの音源（mix）だけを見ると、ボーカルやドラムのノイズに引きずられて和声が揺れる
- 境界時刻が拍・小節とわずかにずれる（再生表示でちらつく）
- ラベルが `C:maj` / `G:maj7` のように冗長で、表示にもログ比較にも不向き

そこで本実装は次の工夫を積み重ねている。

1. **CNN ベース認識器の採用**（`CNNChordFeatureProcessor` + `CRFChordRecognitionProcessor`）
2. **stem 分離を活かした重み付き投票**（核心）
3. **downbeat を最優先する 2 段階スナップ**
4. **ラベル正規化**と**後処理**（同一ラベル結合・短区間統合）

> ⚠️ 用語: 本機能が出力するのは **コード（和音）** であり、**キー（調性）の推定は行っていない**。
> 当初は小節単位のキー（調性）推定モジュールを構想していた（[archive/](archive/) 参照）。
> その「和声 stem を使う」という中核思想が、現行の stem 重み付き投票へと発展している。
> コード列からの調性推定は将来の派生候補。

---

## 2. stem 分離を活かした重み付き投票（核心の工夫）

### 2.1 なぜ stem ごとに認識するのか

Demucs による音源分離（`bass.wav` / `other.wav` / `vocals.wav`）の各 stem に対して、
madmom のコード認識を **個別に** 適用する（[`_run_madmom_multi`](../src/allin1/chord_detection.py)）。
分離音源が無ければオリジナル mix にフォールバックする。

```python
feats = CNNChordFeatureProcessor()(str(stem_path))   # 音声 → CNN コード特徴量（音声を直接受け取る）
output = CRFChordRecognitionProcessor()(feats)        # 特徴量 → CRF → コードラベル列
```

和声情報は主にベース（根音）と other（コード楽器）に乗っており、ボーカルやドラムは
和声判定にとってはノイズに近い。stem ごとに見ることで、ノイズ源を切り分けて評価できる。

### 2.2 重み付き投票によるマージ

全 stem の境界時刻を集めて**統一タイムライン**を作り、その各区間について
各 stem の「その時刻のコード」を集計し、`STEM_WEIGHTS` で重み付き投票する
（[`_merge_stem_results`](../src/allin1/chord_detection.py)）。

| stem | 重み | 役割 |
|---|---|---|
| `bass` | 0.5 | 根音の最有力手掛かり。**ルート音のみ**で投票（`root_only`: `Am`→`A`, `G7`→`G`） |
| `other` | 0.4 | コード楽器（ギター・キーボード等）。コード品質を含めて投票 |
| `vocals` | 0.1 | 補助的。和声判定の主役ではないため小さく |
| `mix` | 1.0 | stem が無いときの単独ソース／primary が N の区間を埋める fallback voter（§2.4） |

**bass をルート音のみで投票する**のがポイント。ベースは根音を強く鳴らすが
三度・七度などの和声色は持たないため、品質まで投票させると誤りやすい。
根音の票として使い、品質は `other` に委ねる、という和声理論に沿った役割分担になっている。

各区間で最も票を集めたラベルを採用し、`ChordSegment(start, end, label)` の列を得る。

### 2.3 stem 無音ゲート（ベース無音区間のフォールバック）

stem が無音でも madmom は `N`(No Chord) を返すため、無条件に投票させると問題が起きる。
特に **ベースが鳴っていないイントロ等** では bass の `N` 票（重み 0.5）が `other` の実コード
（0.4）を outvote し、その区間が永遠に `N` になる（§9.2 の誤りクラス2と同根）。

そこで各区間で **各 stem が実際に鳴っているか** をフレーム RMS で判定し、無音な stem は
その区間の投票から除外する（[`_compute_stem_activity` / `_stem_active`](../src/allin1/chord_detection.py)）。
これにより bass 無音区間では bass 票が捨てられ、自然に `other`/`vocals` が和声を決める。

- 判定は二値（無音なら票を捨てる）。閾値は **stem 自身のピーク RMS に対する相対値**
  （楽曲ごとの音量差に頑健）＋絶対フロア。
- 全 stem が無音の区間は `N`（票が無い）になる。

| 定数 | 値 | 役割 |
|---|---|---|
| `STEM_SILENCE_REL` | 0.10 | 各 stem 自身のピーク RMS に対する相対無音閾値 |
| `STEM_SILENCE_ABS_FLOOR` | 1e-4 | これ未満は常に無音（全体が無音な stem の相対比破綻を防ぐ） |
| `RMS_FRAME_LENGTH` | 2048 | フレーム RMS の窓長 |
| `RMS_HOP_LENGTH` | 512 | フレーム RMS のホップ長 |

### 2.4 mix フォールバック（primary が N の区間を埋める）

無音ゲートを掛けても、**stem は鳴っているのに madmom が `N` を返す**区間が残る。
実測（synth/EDM 系の 2 曲）では、最終出力 N の主因は「全 stem 無音」ではなく
**`other` は鳴っているのに認識器が `N`**（vocals は分離単旋律ゆえ ≈100% が `N` で、
コード源として機能しない）。

そこで `mix`（オリジナル音源）を**常に併走**させ、
**primary（`bass`/`other`/`vocals`）投票の結果が `N` の区間に限り**、鳴っている `mix` の
コードで穴埋めする（[`_merge_stem_results`](../src/allin1/chord_detection.py)）。
分離 stem 単体では `N` でも、全和声が混ざった `mix` ならコードを取れる区間があるため。

- **既存の非 `N` 検出は上書きしない**（穴埋め専用）ため回帰リスクが無い。
- `mix` も無音ゲートの対象（鳴っていなければ穴埋めしない）。
- vocals を「メロディラインのフォールバック源」にする案は棄却。分離 vocals は単旋律で
  和音を持たず、認識器がほぼ全区間 `N` を返すため救済効果が無いことを実測で確認した。
- コスト: madmom の認識を `mix` に対しても 1 回追加で走らせる（おおむね stem 1 本分）。

実測効果（N の時間割合, **DeepChroma 時代**の mix フォールバック単独効果）:
エゴイスティック 64%→36%、マジェスティック 59%→28%。
その後の認識器差し替え（§6）まで含めた**現行スタック全体**では同 2 曲が 64%→8% / 59%→26% まで低下。
残る `N` は `mix` でも認識器が `N` を返す真に困難な区間で、認識器側の限界（§8・§10.2）。

---

## 3. ラベル正規化

madmom の `root:quality` 形式と従来形式の両方に対応して表示用ラベルへ正規化する
（[`normalize_chord`](../src/allin1/chord_detection.py)）。

| 入力 | 正規化後 | 規則 |
|---|---|---|
| `C:maj` | `C` | major は root のみ |
| `A:min` | `Am` | minor は root + `m` |
| `G:maj7` | `G` | 拡張コードは quality を除去 |
| `Bm7` / `Am7` | `Bm` / `Am` | 従来形式の minor |
| `Cmaj7` | `C` | 従来形式の major |
| `N` | `N` | No Chord はそのまま |

正規化は表示用であり、生ラベル（`label_raw`）は保持する（研究用途のため）。
madmom の認識器は Major/Minor の 25 クラス分類のため、dim・aug・sus などは
最近傍の Major/Minor に丸められる。

---

## 4. downbeat を最優先する 2 段階スナップ

コード区間の境界（start / end）を beat / downbeat に吸着させ、再生表示のちらつきと
小節とのズレを抑える（[`snap_to_beats`](../src/allin1/chord_detection.py) →
[`_choose_boundary`](../src/allin1/chord_detection.py)）。

### 4.1 音楽理論的根拠

西洋調性音楽（ポップス・ロック・フォーク・ジャズスタンダード等）では、
コードチェンジは圧倒的に **小節の強拍（downbeat）** で起きる。
4/4 拍子なら beat 1（downbeat）か beat 3 が大半。
madmom の推定器は時間誤差を持つため、生境界が downbeat の近くにあれば、
それはほぼ「downbeat 上のコードチェンジ」を示している。

一方、ジャズ・ファンク・一部のポップスではコードが小節 4 拍目に**先行**する
（次小節の downbeat を先取りする）こともある。そのため downbeat スナップは
**強制ではなく半径制限付き**にして、誤差吸収とシンコペーション保護のバランスを取る。

### 4.2 Phase 1: downbeat 強制スナップ

対象時刻 `t` の `DOWNBEAT_TOL`（0.20 秒）以内にある downbeat を、距離の近い順に試す。
各候補についてハード制約（前後区間との順序・最小区間長）を確認し、最初に通過した候補を採用する。

| side | ハード制約 |
|---|---|
| `start` | `c >= prev_end` かつ `next_boundary - c >= MIN_CHORD_DURATION` |
| `end` | `c - prev_start >= MIN_CHORD_DURATION` かつ `c <= next_boundary` |

旧来のソフトボーナス方式では「より近い beat があると downbeat が無視される」問題があった。
Phase 1 はこれを解消し、半径内の downbeat をスコアの揺れに優先して採用する。

### 4.3 Phase 2: スコアベースフォールバック

Phase 1 で候補が無ければ、`SNAP_WINDOW`（0.35 秒）以内の全 beat / downbeat を候補に集め、
スコアで最適解を選ぶ。

```
score = |c - t|                            # 基本: 元の時刻からの距離
      - DOWNBEAT_BONUS   (downbeat なら)   # downbeat ソフト優遇
      + ORDER_PENALTY*2  (順序違反なら)    # 前後順序ペナルティ（強め）
      + SHORT_SEGMENT_PENALTY (短すぎるなら) # 最小長ペナルティ
      + 0.2              (end が next に近すぎる場合)
```

最小スコアの候補を採用する。候補がなければ元の時刻 `t` を維持する。

### 4.4 パラメータ

| 定数 | 値 | 役割 |
|---|---|---|
| `DOWNBEAT_TOL` | 0.20 s | Phase 1 強制スナップの判定半径 |
| `DOWNBEAT_BONUS` | 0.12 | Phase 2 downbeat ソフト優遇 |
| `SNAP_WINDOW` | 0.35 s | Phase 2 候補収集ウィンドウ |
| `MIN_CHORD_DURATION` | 0.30 s | コード区間の最小長 |
| `ORDER_PENALTY` | 1.0 | 順序違反ペナルティ（×2 して適用） |
| `SHORT_SEGMENT_PENALTY` | 0.5 | 最小長違反ペナルティ |
| `FLOAT_TOL` | 1e-3 | 浮動小数点比較の許容差 |

`DOWNBEAT_TOL = 0.20` の根拠: 一般的なポップス/ロック（60〜180 BPM）の beat 間隔は 0.33〜1.0 s で、
0.20 s はほぼ全 BPM で半拍以内。madmom の時間誤差（概ね 0.05〜0.15 s）を吸収しつつ、
0.25 s 以上前に位置するシンコペーションの誤吸着を避けられる。

---

## 5. 後処理

スナップ後に 2 段階で整える（順序が重要）。

1. **同一ラベル連続区間の結合**（[`_merge_consecutive_same_label`](../src/allin1/chord_detection.py)）
2. **短すぎる区間の統合**（[`_merge_short_segments`](../src/allin1/chord_detection.py)） — `MIN_CHORD_DURATION` 未満を隣接区間に統合（先頭は次へ、それ以外は前へ）。点滅する誤検出を除去する。

---

## 6. 認識器（CNN + CRF）

コード認識は madmom の **CNN ベース認識器**を使う:

- `CNNChordFeatureProcessor` — **音声ファイルを直接受け取り**、CNN でコード特徴量を抽出する。
- `CRFChordRecognitionProcessor` — その特徴量を CRF で復号し、`root:quality` ラベル列を返す。

旧実装は `DeepChromaProcessor` + `DeepChromaChordRecognitionProcessor`（ディープクロマ→CRF）だったが、
A/B 実測（§9）で CNN+CRF が **正解曲で同等以上・難曲で N を大幅削減**だったため差し替えた。

> 補足（撤去した過去の罠）: 旧 DeepChroma の CRF（`ConditionalRandomField`）は numpy 1.x 形式の
> pickle で保存されており、numpy 2.x ではモデル重みが文字列型（`<U*`）として解釈されて
> `_UFuncNoLoopError` でクラッシュした。これを `_patch_madmom_crf` の `process()` モンキーパッチで
> 回避していたが、CNN+CRF 経路では再現しないため、認識器差し替えに伴いパッチごと撤去した。

---

## 7. 出力・使い方

### CLI

```bash
# コード検出あり
uv run allin1 path/to/audio.flac --chords -o ./struct

# 上書き再解析
uv run allin1 path/to/audio.flac --chords --overwrite -o ./struct
```

`--chords` フラグは [`cli.py`](../src/allin1/cli.py) で定義され、`detect_chords` として解析に渡る。
フラグなしの場合 `chords` は `null`。

### データ構造（[`ChordSegment`](../src/allin1/typings.py)）

```python
@dataclass
class ChordSegment:
    start: float        # 開始時刻 (秒)
    end: float          # 終了時刻 (秒)
    label: str          # 正規化済みラベル (例: C, Am)
    label_raw: str = '' # madmom 生出力 (例: C:maj, A:min)
    confidence: float = 0.0
```

### 出力ファイル

保存は [`save_results`](../src/allin1/helpers.py) が担当する。

| ファイル | 内容 |
|---|---|
| `{stem}.json` の `chords` | `{start, end, label, label_raw, confidence}` の配列、または `null` |
| `{stem}_raw_chord.json` | 全 chord segment の生データ（`--chords` 時のみ生成）|

```json
{
  "path": "...",
  "bpm": 120,
  "beats": [ ... ],
  "chords": [
    {"start": 0.0, "end": 2.0, "label": "C",  "label_raw": "C:maj", "confidence": 0.0},
    {"start": 2.0, "end": 4.0, "label": "Am", "label_raw": "A:min", "confidence": 0.0}
  ]
}
```

`N` は madmom の "No Chord" クラス。無音・打楽器のみ・ノイズなど、明確なコードが
検出できない区間に割り当てられる。

---

## 8. 既知の制限

- 検出精度は madmom の CNN 認識器（`CNNChordFeatureProcessor` + `CRFChordRecognitionProcessor`）に依存する
- Major/Minor の 25 クラス分類のため、dim・aug・sus などは最近傍の Major/Minor に丸められる
- CNN 認識を stem 3 本＋mix の計 4 回走らせるため、DeepChroma よりやや計算コストが高い
- `confidence` は現状 madmom から伝播しておらず 0.0 固定（将来拡張余地）
- `--overwrite` なしで既解析済みファイルが存在する場合、コード検出は再実行されない
- キー（調性）推定は未実装
- stem 無音ゲート（§2.3）の相対閾値はフェードイン/アウト境界で投票対象の切替を起こしうるが、
  後段の downbeat スナップ・短区間統合（§4・§5）が境界のガタつきを吸収する

---

## 9. 精度の実測と確定した誤りクラス（`01 - 夜に駆ける`）

`01 - 夜に駆ける` を [04_chord_evaluation.md](04_chord_evaluation.md) の評価基盤で照合した結果。

### 9.0 認識器 A/B 実測（DeepChroma → CNN+CRF 差し替えの根拠）

| 曲 | 指標 | DeepChroma（旧） | CNN+CRF（現行） |
|---|---|---|---|
| `01 - 夜に駆ける`（正解あり） | ルート / 種別 / N / offset | 83% / 92% / 2% / +0 | **83% / 96% / 2% / +0** |
| エゴイスティック・ヒューリスティック | 認識器単体の N 率 | 46% | **9%** |
| マジェスティック・サディスティック | 認識器単体の N 率 | 30% | 29% |

正解曲で**退行なし（ピッチずれ無し・種別はむしろ +4%）**、synth/EDM 系の難曲で **N を激減**。
N 多発の主因が stem 選択でも無音でもなく**認識器の精度**だったことを切り分けた上での差し替え。
（mix fallback・stem 無音ゲートは認識器非依存の改善として併存する。）

### 9.1 「見かけ上の +3 半音ずれ」はカポ表記が原因（実バグではない）

当初 U-FRET のコード譜と比べて検出が一律 +3 半音ずれて見えた（`F→E7→Am7` に対し検出は
`G#→G→C`）。原因は **U-FRET 側がカポ3表記** だったこと。カポ0（原曲キー）に直すと
U-FRET は `A♭→G7→Cm7` となり、**検出のルートと一致する**（A♭=G#, G7=G, Cm7=C）。

評価 CLI の定数オフセット探索でも **best_offset = +0**（グローバルなピッチずれ無し）が出る。
→ リサンプリング/サンプルレート起因の系統的ピッチバグは存在しない。

> ⚠️ リファレンスを写すときは **カポ設定を必ず concert pitch に正規化**すること
> （[04_chord_evaluation.md](04_chord_evaluation.md) のリファレンス形式 `capo` フィールド参照）。

### 9.2 確定した3クラスの誤り（今後の改善ターゲット）

1. **種別 (maj/min) の誤り** — `Cm7` を `C`、`Gm7` を `G` と判定（3度の取り違え）。
   25 クラス分類の限界 ＋ chroma の第3音が弱いと起きる典型誤り。実測 種別精度 ≈ 67%。
2. **コードの欠落・短断片化** — `B♭m7`/`E♭7` 等が検出に現れず、0.46s 程度の短い誤コード
   （`A#`/`C#`/`D#`）に化ける。短断片（<0.5s）混入率 ≈ 19%。疎なボーカル主体の区間（イントロ1）は
   コードが全く立たず `N` になりやすい。
   → うち「**ベース無音区間で bass の `N` 票が other の実コードを潰す**」分は、
   §2.3 の stem 無音ゲートで緩和済み（bass 無音時は other/vocals が和声を決める）。
   → さらに「**stem は鳴っているのに認識器が `N`**」の分は、§2.4 の mix フォールバックで
   一部緩和（synth/EDM 系 2 曲で N 率 64%→36% / 59%→28%）。ただし `mix` でも `N` を返す
   区間は残り、これは認識器側の限界（chroma テンプレ fallback・認識器差し替えが次の打ち手）。
3. **コード数・タイミングのズレ** — 1 フレーズ 4 コードが 3 コードに潰れる（ref 29 vs pred 13）。
   「タイミングがズレている」体感の主因。

> 📈 上記は **DeepChroma 時代の実測**。CNN+CRF 差し替え後（§9.0）の同曲フルパイプライン実測では
> **ルート 41%→79%、種別 67%→74%、コード数 ref29 vs pred 13→27、短断片 19%→10%** と全面的に改善した。
> 特にクラス2の `N` 欠落（イントロ1: ref13→pred1）はほぼ解消（pred13）。
> 残る改善余地は §10 を参照。

これらの改善（認識器の差し替え＝実施済み、短断片除去・小節量子化、語彙拡張）は本評価基盤で
回帰を測りながら進める。

---

## 10. 調査ログ: 棄却した代替案と次の改善候補

> N（No Chord）多発の解消を巡って試した手法の**実測記録**。同じ検証を二度しないために残す。
> 評価は §9.0 と同じ 2 曲（synth/EDM 系）＋正解曲 `01 - 夜に駆ける` で行った。

### 10.1 棄却した代替案（いずれも実測で効果なし）

| 代替案 | 仮説 | 実測結果 | 判定 |
|---|---|---|---|
| **bass+drums ステムにフォールバック** | ドラムを足すとベースが聞き取りやすい | 認識器単体 N が **74%→95% / 73%→93%** と悪化。bass が N の区間の救済 **0%** | 棄却。打楽器は chroma にとってノイズ |
| **bass の打楽器混入を除去**（HPSS-harmonic / lowpass 500Hz） | kick が bass stem に混入して認識を阻害 | HPSS で N **74%→97%**、lowpass はほぼ不変。打楽器を除いても N は減らない | 棄却。混入は主因でない |
| **bass のルート音(ピッチ)で N を穴埋め** | bass の根音で N 区間を埋める | 非 N 区間での一致率（信頼性）が **chroma-argmax / pYIN / 安定区間限定いずれも ~50–58%**（コイントス）。カバレッジは高い(91%/71%)が誤りも半分 | 棄却。誤ラベルを N に書く害が大きい |

**根本理解**: bass stem は**単音（モノフォニック）**で、コード認識器は**三和音(triad)を探す**器。
単音信号には和音が無いため、分離品質に関わらず `N` になりやすい。打楽器を足す/除く・ルート抽出の
いずれも、この**信号と認識器のミスマッチ**を解消しない。N の主因は認識器精度であり、解は
**認識器そのものの差し替え（§6・§9.0 で CNN+CRF を採用）**だった。

### 10.2 次の改善候補（未着手）

- **投票重みによる種別(minor→major)潰し**: パイプライン経由の種別精度（74%）は、CNN を
  mix 単体にかけた値（96%）より低い。仮説は、bass の**ルート専用投票**（`root_only`・重み 0.5・
  品質なし）が other の minor 票（重み 0.4）を上回り、`Am`→`A` のように長短を潰していること。
  投票キーが `"A"`(bass) と `"A:min"`(other) に割れる構造に起因する。重み配分・品質統合の見直しが候補。
- **語彙拡張**: 現状 maj/min 25 クラスのみ。7th/dim/sus は最近傍へ丸められる。
- **短断片除去・小節量子化の強化**: クラス3（コード数・タイミング）の残差対策。

## 関連ドキュメント

- [00_overview.md](00_overview.md) — リポジトリ全体像
- [01_bpm_algorithm.md](01_bpm_algorithm.md) — BPM アルゴリズム
- [04_chord_evaluation.md](04_chord_evaluation.md) — コード精度の評価基盤（`compare_chords.py`）
- [archive/](archive/) — 初期のキー検出設計（現行思想の源流）
- 元設計: [plan/今回のスコープ外.md](../plan/今回のスコープ外.md) — コードレイヤー UI 構想
