"""
検出コード（解析結果JSONの chords）と、U-FRET 由来のリファレンス注釈を照合するスクリプト。

誤りを「ルート / 種別(maj-min) / コード数・タイミング」の3クラスに分解して可視化する。
リファレンスは秒単位の厳密なタイムスタンプを持たないため、各セクション窓内の検出コード列と
リファレンスのコード列を「順序ベース（Needleman-Wunsch アライメント）」で突き合わせる。

Usage:
  uv run python compare_chords.py "music_ml/01 - 夜に駆ける.json" --ref "eval/chord_refs/01 - 夜に駆ける.json"
  uv run python compare_chords.py --struct-dir music/struct --ref-dir eval/chord_refs

リファレンス形式は eval/chord_refs/*.json を参照（capo を必ず明示すること）。
"""
import argparse
import json
import re
import sys
from pathlib import Path
from typing import List, Optional, Tuple

# Windows の cp932 コンソールでも日本語・記号を出力できるよう UTF-8 に揃える。
if hasattr(sys.stdout, 'reconfigure'):
  sys.stdout.reconfigure(encoding='utf-8')

# ルート音 → ピッチクラス(0..11)。シャープ/フラット両対応。
_NOTE_TO_PC = {
  'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3, 'E': 4, 'Fb': 4,
  'E#': 5, 'F': 5, 'F#': 6, 'Gb': 6, 'G': 7, 'G#': 8, 'Ab': 8, 'A': 9,
  'A#': 10, 'Bb': 10, 'B': 11, 'Cb': 11, 'B#': 0,
}
_ROOT_RE = re.compile(r'^([A-G][#b]?)')

# アライメント用の (pitch_class, quality) ペア。pc=None は No-Chord(N)。
ChordTok = Tuple[Optional[int], Optional[str]]

NO_CHORD: ChordTok = (None, None)
MIN_FRAGMENT = 0.5  # この秒数未満の検出区間は「短断片ノイズ」として扱う


def parse_chord(label: str) -> ChordTok:
  """コードラベルを (pitch_class, quality) に変換する。

  種別判定は src/allin1/chord_detection.py の normalize_chord と整合させる:
    C/C#/Ab/G7/Cmaj7 → maj, Am/Cm7/Bbm7 → min, C:min → min, N/X/'' → No-Chord
  """
  if label is None:
    return NO_CHORD
  label = label.strip()
  if label in ('N', 'X', ''):
    return NO_CHORD
  m = _ROOT_RE.match(label)
  if not m:
    return NO_CHORD
  root = m.group(1)
  pc = _NOTE_TO_PC.get(root)
  if pc is None:
    return NO_CHORD
  rest = label[len(root):]
  # madmom の "root:quality" 形式
  if rest.startswith(':'):
    quality = 'min' if rest[1:].startswith('min') else 'maj'
  elif rest.startswith('m') and not rest.startswith('maj'):
    quality = 'min'
  else:
    quality = 'maj'
  return (pc, quality)


def tok_str(tok: ChordTok, offset: int = 0) -> str:
  pc, quality = tok
  if pc is None:
    return 'N'
  names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
  name = names[(pc + offset) % 12]
  return name + ('m' if quality == 'min' else '')


def transpose(tok: ChordTok, offset: int) -> ChordTok:
  pc, quality = tok
  if pc is None:
    return tok
  return ((pc + offset) % 12, quality)


def collect_predicted(chords: List[dict], start: float, end: float):
  """窓 [start, end) に中心がある検出区間を返す。"""
  out = []
  for c in chords:
    s, e = float(c['start']), float(c['end'])
    center = (s + e) / 2.0
    if start <= center < end:
      out.append((s, e, parse_chord(c.get('label', 'N'))))
  return out


def clean_sequence(segs):
  """短断片(<MIN_FRAGMENT)を除去し、連続同一トークンを結合した列を返す。"""
  kept = [(s, e, t) for (s, e, t) in segs if (e - s) >= MIN_FRAGMENT]
  merged: List[ChordTok] = []
  for _, _, t in kept:
    if not merged or merged[-1] != t:
      merged.append(t)
  return merged


def align(ref: List[ChordTok], pred: List[ChordTok],
          match=1, mismatch=-1, gap=-1):
  """ルート(pitch class)に基づく Needleman-Wunsch 大域アライメント。

  返り値: [(ref_tok|None, pred_tok|None), ...]
  """
  n, m = len(ref), len(pred)
  dp = [[0] * (m + 1) for _ in range(n + 1)]
  for i in range(1, n + 1):
    dp[i][0] = i * gap
  for j in range(1, m + 1):
    dp[0][j] = j * gap

  def sub(a: ChordTok, b: ChordTok) -> int:
    if a[0] is not None and a[0] == b[0]:
      return match
    return mismatch

  for i in range(1, n + 1):
    for j in range(1, m + 1):
      dp[i][j] = max(
        dp[i - 1][j - 1] + sub(ref[i - 1], pred[j - 1]),
        dp[i - 1][j] + gap,
        dp[i][j - 1] + gap,
      )

  # トレースバック
  i, j = n, m
  out = []
  while i > 0 or j > 0:
    if i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + sub(ref[i - 1], pred[j - 1]):
      out.append((ref[i - 1], pred[j - 1]))
      i, j = i - 1, j - 1
    elif i > 0 and dp[i][j] == dp[i - 1][j] + gap:
      out.append((ref[i - 1], None))
      i -= 1
    else:
      out.append((None, pred[j - 1]))
      j -= 1
  out.reverse()
  return out


def evaluate_track(data: dict, ref: dict, offset: int = 0, verbose: bool = True):
  """1曲を評価して集計 dict を返す。offset はリファレンスへ加える定数移調(半音)。"""
  capo = int(ref.get('capo', 0))
  chords = data.get('chords') or []

  agg = {
    'ref_total': 0, 'root_match': 0, 'quality_match': 0,
    'pred_total_clean': 0, 'short_frag': 0, 'pred_raw': 0,
  }

  for sec in ref.get('sections', []):
    s = float(sec['approx_start'])
    e = float(sec.get('approx_end', s + 1e9))
    raw = collect_predicted(chords, s, e)
    pred_seq = clean_sequence(raw)
    ref_seq = [transpose(parse_chord(c), capo + offset) for c in sec.get('chords', [])]

    short = sum(1 for (a, b, _) in raw if (b - a) < MIN_FRAGMENT)
    agg['pred_raw'] += len(raw)
    agg['short_frag'] += short
    agg['pred_total_clean'] += len(pred_seq)
    agg['ref_total'] += len(ref_seq)

    pairs = align(ref_seq, pred_seq)
    sec_root = sec_qual = 0
    for r, p in pairs:
      if r is not None and p is not None and r[0] is not None and r[0] == p[0]:
        sec_root += 1
        if r[1] == p[1]:
          sec_qual += 1
    agg['root_match'] += sec_root
    agg['quality_match'] += sec_qual

    if verbose and offset == 0:
      print(f"\n  [{sec.get('label', '?')}] {s:.2f}-{e:.2f}s  "
            f"ref={len(ref_seq)} pred={len(pred_seq)} (raw {len(raw)}, 短断片 {short})")
      ref_line = ' '.join(tok_str(r) if r else '-' for r, _ in pairs)
      prd_line = ' '.join(tok_str(p) if p else '-' for _, p in pairs)
      mark_line = ' '.join(
        ('=' if (r and p and r[0] == p[0] and r[1] == p[1])
         else ('~' if (r and p and r[0] is not None and r[0] == p[0]) else 'x'))
        .center(max(len(tok_str(r) if r else '-'), len(tok_str(p) if p else '-')))
        for r, p in pairs
      )
      print(f"    ref : {ref_line}")
      print(f"    pred: {prd_line}")
      print(f"    判定: {mark_line}")

  return agg


def find_best_offset(data: dict, ref: dict) -> Tuple[int, int]:
  """ルート一致が最大になる定数移調オフセットを返す (best_offset, best_root_match)。"""
  best_off, best_score = 0, -1
  for off in range(12):
    a = evaluate_track(data, ref, offset=off, verbose=False)
    if a['root_match'] > best_score:
      best_score, best_off = a['root_match'], off
  return best_off, best_score


def report(name: str, data: dict, ref: dict):
  print(f"\n=== {name} (source={ref.get('source', '?')}, capo={ref.get('capo', 0)}) ===")
  agg = evaluate_track(data, ref, offset=0, verbose=True)

  rt = agg['ref_total'] or 1
  rm = agg['root_match']
  qm = agg['quality_match']
  print('\n  --- メトリクス ---')
  print(f"  ルート精度   : {rm}/{rt} ({rm / rt * 100:.1f}%)")
  print(f"  種別(maj/min): {qm}/{rm if rm else 1} ({qm / (rm or 1) * 100:.1f}%)  ※ルート一致のうち")
  print(f"  コード数     : ref={agg['ref_total']} vs pred(clean)={agg['pred_total_clean']} "
        f"(差 {agg['pred_total_clean'] - agg['ref_total']:+d})")
  print(f"  短断片混入   : {agg['short_frag']}/{agg['pred_raw']} 区間 "
        f"({agg['short_frag'] / (agg['pred_raw'] or 1) * 100:.1f}% が <{MIN_FRAGMENT}s)")

  best_off, _ = find_best_offset(data, ref)
  status = 'OK (ピッチずれ無し)' if best_off == 0 else f'要注意: 定数 {best_off:+d} 半音ずれ'
  print(f"  best_offset  : {best_off:+d} 半音  -> {status}")
  return agg


def main():
  parser = argparse.ArgumentParser(description='検出コード vs リファレンス 照合スクリプト')
  parser.add_argument('analysis', nargs='?', type=Path, help='解析結果JSON（単一ファイルモード）')
  parser.add_argument('--ref', type=Path, help='リファレンス注釈JSON（単一ファイルモード）')
  parser.add_argument('--struct-dir', type=Path, help='解析結果JSONのディレクトリ（バッチモード）')
  parser.add_argument('--ref-dir', type=Path, default=Path('eval/chord_refs'),
                      help='リファレンス注釈のディレクトリ（バッチモード, default: eval/chord_refs）')
  args = parser.parse_args()

  pairs = []
  if args.analysis:
    ref_path = args.ref or (args.ref_dir / f'{args.analysis.stem}.json')
    if not ref_path.exists():
      parser.error(f'リファレンスが見つかりません: {ref_path}')
    pairs.append((args.analysis.stem, args.analysis, ref_path))
  elif args.struct_dir:
    for ref_path in sorted(args.ref_dir.glob('*.json')):
      struct = args.struct_dir / f'{ref_path.stem}.json'
      if struct.exists():
        pairs.append((ref_path.stem, struct, ref_path))
      else:
        print(f'[WARN] 解析結果が無い: {struct}')
  else:
    parser.error('analysis を指定するか、--struct-dir でバッチ指定してください。')

  for name, struct, ref_path in pairs:
    data = json.loads(struct.read_text(encoding='utf-8'))
    ref = json.loads(ref_path.read_text(encoding='utf-8'))
    report(name, data, ref)


if __name__ == '__main__':
  main()
