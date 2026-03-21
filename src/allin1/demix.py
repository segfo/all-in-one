import sys
import subprocess
import torch

from pathlib import Path
from typing import List, Union


def demix(paths: List[Path], demix_dir: Path, device: Union[str, torch.device]):
  """Demixes the audio file into its sources."""
  todos = []
  demix_paths = []
  for path in paths:
    out_dir = demix_dir / 'htdemucs' / path.stem
    demix_paths.append(out_dir)
    if out_dir.is_dir():
      if (
        (out_dir / 'bass.wav').is_file() and
        (out_dir / 'drums.wav').is_file() and
        (out_dir / 'other.wav').is_file() and
        (out_dir / 'vocals.wav').is_file()
      ):
        continue
    todos.append(path)

  existing = len(paths) - len(todos)
  print(f'=> Found {existing} tracks already demixed, {len(todos)} to demix.')

  if todos:
    # allin1.demucs_runner 経由で起動することで、demucs サブプロセスにも
    # ROCm パッチ（Flash/Mem-Efficient SDP 無効化等）が適用される。
    subprocess.run(
      [
        sys.executable, '-m', 'allin1.demucs_runner',
        '--out', demix_dir.as_posix(),
        '--name', 'htdemucs',
        '--device', str(device),
        *[path.as_posix() for path in todos],
      ],
      check=True,
    )

  return demix_paths
