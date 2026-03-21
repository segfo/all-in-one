"""
HumanBPM ディレクトリの人手BPM値と、分析結果JSONのBPMを比較するスクリプト。

Usage:
  python compare_bpm.py
  uv run compare_bpm.py --struct-dir music/struct --human-dir HumanBPM
"""
import argparse
import json
from pathlib import Path


def main():
  parser = argparse.ArgumentParser(description='HumanBPM vs 推定BPM 照合スクリプト')
  parser.add_argument('--struct-dir', type=Path, default=Path('music/struct'),
                      help='分析結果JSONのディレクトリ (default: music/struct)')
  parser.add_argument('--human-dir', type=Path, default=Path('HumanBPM'),
                      help='人手BPM txtファイルのディレクトリ (default: HumanBPM)')
  args = parser.parse_args()

  records = []
  for human_file in sorted(args.human_dir.glob('*.txt')):
    try:
      human_bpm = int(human_file.read_text(encoding='utf-8').strip())
    except ValueError:
      print(f'[WARN] {human_file.name}: BPM値の解析に失敗')
      continue

    stem = human_file.stem
    json_file = args.struct_dir / f'{stem}.json'
    if not json_file.exists():
      records.append({'name': stem, 'human': human_bpm, 'computed': None, 'diff': None})
      continue

    data = json.loads(json_file.read_text(encoding='utf-8'))
    computed_bpm = data.get('bpm')
    diff = abs(computed_bpm - human_bpm) if computed_bpm is not None else None
    records.append({
      'name': stem,
      'human': human_bpm,
      'computed': computed_bpm,
      'diff': diff,
    })

  if not records:
    print('照合対象が見つかりませんでした。')
    return

  name_w = max(len(r['name']) for r in records) + 2
  name_w = max(name_w, 12)

  header = f"{'曲名':<{name_w}} {'人手BPM':>8} {'推定BPM':>8} {'差':>6}  判定"
  print(header)
  print('-' * (name_w + 32))

  total, within_2, exact = 0, 0, 0
  for r in records:
    diff_str = f'{r["diff"]:+d}' if r['diff'] is not None else 'N/A'
    computed_str = str(r['computed']) if r['computed'] is not None else '未分析'

    if r['diff'] is None:
      mark = '-'
    elif r['diff'] == 0:
      mark = 'OK  exact'
      exact += 1
      within_2 += 1
      total += 1
    elif r['diff'] <= 2:
      mark = 'OK  +-2BPM'
      within_2 += 1
      total += 1
    else:
      mark = 'NG'
      total += 1

    print(f'{r["name"]:<{name_w}} {r["human"]:>8} {computed_str:>8} {diff_str:>6}  {mark}')

  print('-' * (name_w + 32))
  if total:
    print(f'\nexact match : {exact}/{total} ({exact/total*100:.1f}%)')
    print(f'within +-2  : {within_2}/{total} ({within_2/total*100:.1f}%)')
    unanalyzed = sum(1 for r in records if r['diff'] is None)
    if unanalyzed:
      print(f'unanalyzed  : {unanalyzed}')


if __name__ == '__main__':
  main()
