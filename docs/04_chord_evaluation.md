# 04. コード精度の評価基盤

> コード検出（[02_chord_detection.md](02_chord_detection.md)）の精度を、リファレンス（人手のコード譜）と
> 定量比較するための仕組み。改善の効果を「測りながら」進めるために用意した。

正本スクリプト: [`compare_chords.py`](../compare_chords.py)（リポジトリ直下。BPM 用の
[`compare_bpm.py`](../compare_bpm.py) と同じ規約）。

---

## 1. 何を測るか — 誤りを3クラスに分解する

検出コードの誤りは性質が異なるため、一つの一致率に潰さず分けて出す。

| メトリクス | 意味 | 何が分かるか |
|---|---|---|
| **ルート精度** | ルート音（pitch class）のみの一致率 | ピッチ自体が合っているか。低ければ移調/ピッチ系の疑い |
| **種別 (maj/min)** | ルート一致時に長短も合う割合 | 3 度の取り違え（例 `Cm7`→`C`） |
| **コード数** | ref と検出（クリーン後）の個数差 | フレーズのコードが潰れていないか／余計に増えていないか |
| **短断片混入** | `<0.5s` の検出区間の割合 | 点滅する誤コードノイズの量 |
| **best_offset** | ルート一致が最大になる定数移調(半音) | **0 が正常**。0 以外なら系統的なピッチ/カポずれ |

`best_offset` は退行検知にも使える（将来サンプルレート等の事故で全体がずれたら 0 以外になる）。

---

## 2. 比較のしくみ

リファレンス（U-FRET 等）は秒単位の厳密なタイムスタンプを持たないため、**秒ではなく順序で**突き合わせる。

1. 各セクション窓 `[approx_start, approx_end)` に中心がある検出区間を集める。
2. `<0.5s` の短断片を除去し、連続同一コードを結合して代表列にする。
3. リファレンス列に **`capo`（＋探索オフセット）を適用して concert pitch に正規化**。
4. ルート（pitch class）に基づく **Needleman-Wunsch 大域アライメント**で対応付け。
5. 対応位置からルート/種別の一致数、欠落（gap）を集計。

ラベル解釈は [`src/allin1/chord_detection.py`](../src/allin1/chord_detection.py) の
`normalize_chord` / `root_only` と整合させてある（`Cm7`→min, `G7`/`Ab`→maj, `N`→無コード）。

---

## 3. リファレンス注釈フォーマット

`eval/chord_refs/<曲名>.json`。例: [`eval/chord_refs/01 - 夜に駆ける.json`](../eval/chord_refs/01%20-%20夜に駆ける.json)

```json
{
  "track": "01 - 夜に駆ける",
  "source": "U-FRET",
  "capo": 0,
  "key_hint": "Cm",
  "sections": [
    { "label": "intro2 (inst)", "approx_start": 16.99, "approx_end": 31.76,
      "chords": ["Ab", "G7", "Cm7", "Ab", "G7", "Cm7", "Bbm7", "Eb7"] }
  ]
}
```

- **`capo` は必須級**。U-FRET はカポ表記で見せることがあり、`capo=3` のまま写すと実音と 3 半音ずれる。
  写したカポ値をそのまま `capo` に入れれば、スクリプトが `label + capo` 半音で concert pitch に直す。
  カポ0（原曲キー）に直してから写すなら `capo: 0`。
- `chords` は順序ベースで比較されるので、小節境界に厳密でなくても良い（コード数の差として現れる）。
- `approx_start` / `approx_end` は解析結果 JSON の `segments` の境界を流用すると楽。

---

## 4. 使い方

```bash
# 単一ファイル（ref を明示）
uv run python compare_chords.py "music_ml/01 - 夜に駆ける.json" --ref "eval/chord_refs/01 - 夜に駆ける.json"

# 単一ファイル（eval/chord_refs/<stem>.json を自動解決）
uv run python compare_chords.py "music_ml/01 - 夜に駆ける.json"

# バッチ（解析結果ディレクトリ × リファレンスディレクトリ、stem で対応）
uv run python compare_chords.py --struct-dir music_ml --ref-dir eval/chord_refs
```

解析結果 JSON は `chords`（`{start,end,label}` の配列）を含むものなら allin1 出力でも
MusicAnalyzer 出力でも可。

### 出力の読み方（実測例）

```
  [intro2 (inst)] 16.99-31.76s  ref=16 pred=12 (raw 15, 短断片 3)
    ref : G# G Cm G# G Cm A#m ...
    pred: G# G C  G# G C  -   ...
    判定: =  = ~  =  = ~  x   ...
  --- メトリクス ---
  ルート精度   : 12/29 (41.4%)
  種別(maj/min): 8/12 (66.7%)
  コード数     : ref=29 vs pred(clean)=13 (差 -16)
  短断片混入   : 3/16 区間 (18.8% が <0.5s)
  best_offset  : +0 半音  -> OK (ピッチずれ無し)
```

`=` ルート＋種別一致 / `~` ルートのみ一致（種別違い）/ `x` 不一致 or 欠落。
上例は「ピッチは合うが（best_offset=0, ルート一致）、種別違い・コード潰れ・短断片」という
[02 §9.2](02_chord_detection.md#92-確定した3クラスの誤り今後の改善ターゲット) の診断を定量化したもの。

---

## 関連ドキュメント

- [02_chord_detection.md](02_chord_detection.md) — コード検出の設計と既知の誤りクラス
- [00_overview.md](00_overview.md) — リポジトリ全体像
