"""demucs サブプロセス用エントリポイント。

ROCm パッチを適用してから demucs.separate を実行する。
demix.py から `-m allin1.demucs_runner` として呼び出される。

理由: demucs は別プロセスで動作するため、メインプロセスの torch.backends 設定が
引き継がれない。このラッパーを経由することで HTDemucs の cross-attention でも
ROCm の Flash/Mem-Efficient SDP NaN バグを回避できる。
"""

from allin1.rocm_patch import apply_rocm_patches, patch_torchaudio_load

apply_rocm_patches()
patch_torchaudio_load()

# python -m demucs.separate と同等の実行
import runpy  # noqa: E402
runpy.run_module('demucs.separate', run_name='__main__')
