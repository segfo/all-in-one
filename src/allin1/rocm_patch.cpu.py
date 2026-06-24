"""ROCm (AMD GPU / HIP) 固有のバグ回避パッチ。(CPU版)

ドキュメント参照：ROCm 対応記録_Allinone 用/CUDAtoROCmPorting.md
テスト用、本番はrocm_patch.pyで実行する。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

# ROCm 環境で使用する NATTEN バックエンド。
# auto 選択が効かないため flex-fna を明示指定する（CUDAtoROCmPorting.md #NATTEN）。
NATTEN_BACKEND_ROCM: str = 'flex-fna'


def is_rocm() -> bool:
    """ROCm (HIP) 環境かどうかを返す。"""
    return getattr(torch.version, 'hip', None) is not None


def apply_rocm_patches() -> None:
    """ROCm 環境でのみ既知バグの回避パッチを適用する。

    非 ROCm 環境（CUDA/CPU）では何もしない。
    CUDAtoROCmPorting.md の診断チェックリストに基づき以下を適用：
      - Flash/Mem-Efficient SDP 無効化 (Bug #4)
      - LayerNorm CPU フォールバック (Bug #1)
      - GroupNorm CPU フォールバック (Bug #2)
      - demucs sin_embedding CPU 計算 (Bug #3)
      - Conv1d CPU フォールバック (Bug #5)
      - demucs htdemucs mean/std CPU フォールバック (Bug #16)
      - demucs STFT/iSTFT CPU フォールバック (Bug #15)
    """
    if not is_rocm():
        return

    # Bug #4: Flash SDP / Mem-Efficient SDP が NaN を生成する
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)

    # Bug #1: LayerNorm 大規模 reduction NaN（CPU フォールバックで回避）
    _patch_layer_norm()

    # Bug #2: GroupNorm 大規模 reduction NaN（CPU フォールバックで回避）
    _patch_group_norm()

    # Bug #3: demucs sin_embedding の整数テンソルべき乗が NaN を生成
    _patch_sin_embedding()

    # Bug #5: fp16 Conv1d NaN（CPU フォールバックで回避）
    _patch_conv1d()

    # Bug #16: demucs htdemucs mean/std reduction NaN（CPU フォールバックで回避）
    _patch_htdemucs_normalization()

    # Bug #15: demucs STFT/iSTFT CPU フォールバック（fp16 のみ fp32 アップキャストから完全 CPU 転送に変更）
    _patch_stft_iSTFT()


def _patch_layer_norm() -> None:
    """nn.LayerNorm の forward を ROCm NaN セーフ版に差し替える。

    ROCm の GPU カーネルは大規模 reduction で NaN を生成する (CUDAtoROCmPorting.md Bug #1)。
    回避策として CPU に転送して F.layer_norm を実行し、結果を GPU に戻す。
    PCIe 転送が発生するため推論速度は低下するが、NaN バグを確実に回避できる。

    数式:
        out = (x - mean) / sqrt(var + eps) * weight + bias
    """
    if getattr(nn.LayerNorm, '_rocm_patched', False):
        return

    _orig_forward = nn.LayerNorm.forward

    def _rocm_forward(self, x: torch.Tensor) -> torch.Tensor:
        if not (x.is_cuda and torch.version.hip is not None):
            return _orig_forward(self, x)
        # CPU に転送して LayerNorm を実行（ROCm の reduction NaN バグを回避）
        x_cpu = x.cpu()
        weight_cpu = self.weight.cpu() if self.weight is not None else None
        bias_cpu = self.bias.cpu() if self.bias is not None else None
        out_cpu = F.layer_norm(
            x_cpu,
            self.normalized_shape,
            weight_cpu,
            bias_cpu,
            self.eps,
        )
        return out_cpu.to(x.device)

    nn.LayerNorm.forward = _rocm_forward
    nn.LayerNorm._rocm_patched = True


def _patch_group_norm() -> None:
    """nn.GroupNorm の forward を ROCm NaN セーフ版に差し替える。

    ROCm の group_norm カーネルが大規模 reduction で NaN を生成する (CUDAtoROCmPorting.md Bug #2)。
    回避策として CPU に転送して F.group_norm を実行し、結果を GPU に戻す。
    PCIe 転送が発生するため推論速度は低下するが、NaN バグを確実に回避できる。

    数式:
        x_grouped = x.view(B, G, -1)           # [B, G, N]  N = C//G * spatial
        mean = x_grouped.mean(dim=-1)           # [B, G]
        var  = x_grouped.var(dim=-1, unbiased=False)
        x_norm = (x - mean) / sqrt(var + eps)
        out  = x_norm * weight + bias

    Args:
        num_groups: グループ数 G（C の約数）
        weight:     スケール係数 [C] または None
        bias:       バイアス [C] または None
        eps:        数値安定化定数
    """
    if getattr(nn.GroupNorm, '_rocm_patched', False):
        return

    _orig_forward = nn.GroupNorm.forward

    def _rocm_forward(self, x: torch.Tensor) -> torch.Tensor:
        if not (x.is_cuda and torch.version.hip is not None):
            return _orig_forward(self, x)
        # CPU に転送して GroupNorm を実行（ROCm の reduction NaN バグを回避）
        x_cpu = x.cpu()
        weight_cpu = self.weight.cpu() if self.weight is not None else None
        bias_cpu = self.bias.cpu() if self.bias is not None else None
        out_cpu = F.group_norm(
            x_cpu,
            self.num_groups,
            weight_cpu,
            bias_cpu,
            self.eps,
        )
        return out_cpu.to(x.device)

    nn.GroupNorm.forward = _rocm_forward
    nn.GroupNorm._rocm_patched = True


def _patch_sin_embedding() -> None:
    """demucs の create_sin_embedding を CPU 計算に差し替える。

    整数テンソルへのべき乗演算 (max_period ** (adim / (half_dim - 1))) が
    ROCm GPU 上で NaN を生成し、demix 結果の wav が全サンプル -1.0 になる問題を回避する。
    """
    try:
        import demucs.transformer as _dt
    except ImportError:
        return

    if getattr(_dt, '_rocm_sin_patched', False):
        return

    _orig = _dt.create_sin_embedding

    def _rocm_create_sin_embedding(length, dim, shift=0, device='cpu', max_period=10000):
        _dev = device if isinstance(device, torch.device) else torch.device(str(device))
        if _dev.type == 'cuda' and getattr(torch.version, 'hip', None) is not None:
            result_cpu = _orig(length, dim, shift=shift, device='cpu', max_period=max_period)
            return result_cpu.to(_dev)
        return _orig(length, dim, shift=shift, device=device, max_period=max_period)

    _dt.create_sin_embedding = _rocm_create_sin_embedding
    _dt._rocm_sin_patched = True


def _patch_conv1d() -> None:
    """nn.Conv1d の forward を ROCm NaN セーフ版に差し替える。

    ROCm の fp16 Conv1d カーネルは数値的バグにより NaN を生成する (CUDAtoROCmPorting.md Bug #5)。
    回避策として CPU に転送して F.conv1d を実行し、結果を GPU に戻す。
    PCIe 転送が発生するため推論速度は低下するが、NaN バグを確実に回避できる。

    Args:
        x:          入力テンソル [B, C_in, L]
        weight:     重み [C_out, C_in, kernel_size]
        bias:       バイアス [C_out] または None
    """
    if getattr(nn.Conv1d, '_rocm_patched', False):
        return

    _orig_forward = nn.Conv1d.forward

    def _rocm_forward(self, x: torch.Tensor) -> torch.Tensor:
        if not (x.is_cuda and torch.version.hip is not None and x.dtype == torch.float16):
            return _orig_forward(self, x)
        # CPU に転送して Conv1d を実行（ROCm の fp16 NaN バグを回避）
        x_cpu = x.cpu()
        weight_cpu = self.weight.cpu()
        bias_cpu = self.bias.cpu() if self.bias is not None else None
        out_cpu = F.conv1d(
            x_cpu,
            weight_cpu,
            bias_cpu,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
        return out_cpu.to(x.device)

    nn.Conv1d.forward = _rocm_forward
    nn.Conv1d._rocm_patched = True


def _patch_htdemucs_normalization() -> None:
    """HTDemucs の forward 内の mean/std 正規化を CPU フォールバックに差し替える。

    ROCm の大規模 reduction 演算が NaN を生成する (CUDAtoROCmPorting.md Bug #16)。
    回避策として CPU に転送して mean/std を計算し、結果を GPU に戻す。

    対象箇所：htdemucs.py の forward() 内、正規化ステップ 2 箇所
      - 周波数ブランチ：mean = x.mean(dim=(1, 2, 3)), std = x.std(dim=(1, 2, 3))
      - 時間ブランチ：meant = xt.mean(dim=(1, 2)), stdt = xt.std(dim=(1, 2))

    元の forward メソッドの構造を完全に維持するため、全コードをコピーして
    mean/std の計算部分だけを CPU フォールバックに変更する。
    """
    try:
        import demucs.htdemucs as _htd
    except ImportError:
        return

    if getattr(_htd.HTDemucs, '_rocm_norm_patched', False):
        return

    _orig_forward = _htd.HTDemucs.forward

    def _rocm_forward(self, mix):
        # 元の forward の先頭部分をそのままコピー（length_pre_pad の処理を含む）
        length = mix.shape[-1]
        length_pre_pad = None
        if self.use_train_segment:
            if self.training:
                from fractions import Fraction
                self.segment = Fraction(mix.shape[-1], self.samplerate)
            else:
                training_length = int(self.segment * self.samplerate)
                if mix.shape[-1] < training_length:
                    length_pre_pad = mix.shape[-1]
                    mix = F.pad(mix, (0, training_length - length_pre_pad))

        z = self._spec(mix)
        mag = self._magnitude(z).to(mix.device)
        x = mag

        B, C, Fq, T = x.shape

        # unlike previous Demucs, we always normalize because it is easier.
        # ROCm CPU フォールバック：GPU 上の reduction NaN バグを回避するため CPU で計算
        _is_rocm = x.is_cuda and torch.version.hip is not None

        if _is_rocm:
            x_cpu = x.cpu()
            mean = x_cpu.mean(dim=(1, 2, 3), keepdim=True).to(x.device)
            std = x_cpu.std(dim=(1, 2, 3), keepdim=True).to(x.device)
        else:
            mean = x.mean(dim=(1, 2, 3), keepdim=True)
            std = x.std(dim=(1, 2, 3), keepdim=True)
        x = (x - mean) / (1e-5 + std)

        # x will be the freq. branch input.

        # Prepare the time branch input.
        xt = mix
        if _is_rocm:
            xt_cpu = xt.cpu()
            meant = xt_cpu.mean(dim=(1, 2), keepdim=True).to(xt.device)
            stdt = xt_cpu.std(dim=(1, 2), keepdim=True).to(xt.device)
        else:
            meant = xt.mean(dim=(1, 2), keepdim=True)
            stdt = xt.std(dim=(1, 2), keepdim=True)
        xt = (xt - meant) / (1e-5 + stdt)

        # okay, this is a giant mess I know...
        saved = []  # skip connections, freq.
        saved_t = []  # skip connections, time.
        lengths = []  # saved lengths to properly remove padding, freq branch.
        lengths_t = []  # saved lengths for time branch.

        for idx, encode in enumerate(self.encoder):
            lengths.append(x.shape[-1])
            inject = None
            if idx < len(self.tencoder):
                # we have not yet merged branches.
                lengths_t.append(xt.shape[-1])
                tenc = self.tencoder[idx]
                xt = tenc(xt)
                if not tenc.empty:
                    # save for skip connection
                    saved_t.append(xt)
                else:
                    # tenc contains just the first conv., so that now time and freq.
                    # branches have the same shape and can be merged.
                    inject = xt
            x = encode(x, inject)
            if idx == 0 and self.freq_emb is not None:
                # add frequency embedding to allow for non equivariant convolutions
                # over the frequency axis.
                frs = torch.arange(x.shape[-2], device=x.device)
                emb = self.freq_emb(frs).t()[None, :, :, None].expand_as(x)
                x = x + self.freq_emb_scale * emb

            saved.append(x)

        if self.crosstransformer:
            if self.bottom_channels:
                b, c, f, t = x.shape
                x = self.channel_upsampler(x.reshape(b, c, f * t))
                x = x.reshape(b, -1, f, t)
                xt = self.channel_upsampler_t(xt)

            x, xt = self.crosstransformer(x, xt)

            if self.bottom_channels:
                x = self.channel_downsampler(x.reshape(b, -1, f * t))
                x = x.reshape(b, -1, f, t)
                xt = self.channel_downsampler_t(xt)

        for idx, decode in enumerate(self.decoder):
            skip = saved.pop(-1)
            x, pre = decode(x, skip, lengths.pop(-1))
            # `pre` contains the output just before final transposed convolution,
            # which is used when the freq. and time branch separate.

            offset = self.depth - len(self.tdecoder)
            if idx >= offset:
                tdec = self.tdecoder[idx - offset]
                length_t = lengths_t.pop(-1)
                if tdec.empty:
                    assert pre.shape[2] == 1, pre.shape
                    pre = pre[:, :, 0]
                    xt, _ = tdec(pre, None, length_t)
                else:
                    skip = saved_t.pop(-1)
                    xt, _ = tdec(xt, skip, length_t)

        # Let's make sure we used all stored skip connections.
        assert len(saved) == 0
        assert len(lengths_t) == 0
        assert len(saved_t) == 0

        S = len(self.sources)
        x = x.view(B, S, -1, Fq, T)
        x = x * std[:, None] + mean[:, None]

        # to cpu as mps doesnt support complex numbers
        # demucs issue #435 ##432
        # NOTE: in this case z already is on cpu
        # TODO: remove this when mps supports complex numbers
        x_is_mps = x.device.type == "mps"
        if x_is_mps:
            x = x.cpu()

        zout = self._mask(z, x)
        if self.use_train_segment:
            if self.training:
                x = self._ispec(zout, length)
            else:
                x = self._ispec(zout, training_length)
        else:
            x = self._ispec(zout, length)

        # back to mps device
        if x_is_mps:
            x = x.to("mps")

        if self.use_train_segment:
            if self.training:
                xt = xt.view(B, S, -1, length)
            else:
                xt = xt.view(B, S, -1, training_length)
        else:
            xt = xt.view(B, S, -1, length)
        xt = xt * stdt[:, None] + meant[:, None]
        x = xt + x

        # 元の forward から追加：length_pre_pad の処理（パディング除去）
        if length_pre_pad is not None:
            x = x[..., :length_pre_pad]

        return x

    _htd.HTDemucs.forward = _rocm_forward
    _htd.HTDemucs._rocm_norm_patched = True


def _patch_stft_iSTFT() -> None:
    """demucs.spec の spectro/ispectro を CPU フォールバックに差し替える。

    ROCm の STFT/iSTFT カーネルが fp16 入力で精度問題を起こす場合がある (CUDAtoROCmPorting.md Bug #15)。
    回避策として CPU に転送して計算し、結果を GPU に戻す。

    対象：spec.py の spectro() と ispectro() 関数
    """
    try:
        import demucs.spec as _spec
    except ImportError:
        return

    if getattr(_spec, '_rocm_stft_patched', False):
        return

    # 元の関数を保存
    _orig_spectro = _spec.spectro
    _orig_ispectro = _spec.ispectro

    def _rocm_spectro(x, n_fft=512, hop_length=None, pad=0):
        """CPU フォールバック版 STFT。"""
        *other, length = x.shape
        x = x.reshape(-1, length)

        # ROCm 環境では CPU に転送して計算
        _is_rocm = x.is_cuda and torch.version.hip is not None

        if _is_rocm:
            # CPU で STFT を実行
            x_cpu = x.cpu()
            z = torch.stft(
                x_cpu,
                n_fft * (1 + pad),
                hop_length or n_fft // 4,
                window=torch.hann_window(n_fft).to(x_cpu),
                win_length=n_fft,
                normalized=True,
                center=True,
                return_complex=True,
                pad_mode='reflect',
            )
            _, freqs, frame = z.shape
            result = z.view(*other, freqs, frame)
            return result
        else:
            # 非 ROCm 環境：元の関数をそのまま使用
            return _orig_spectro(x, n_fft, hop_length, pad)

    def _rocm_ispectro(z, hop_length=None, length=None, pad=0):
        """CPU フォールバック版 iSTFT。"""
        *other, freqs, frames = z.shape
        n_fft = 2 * freqs - 2
        z = z.view(-1, freqs, frames)

        # ROCm 環境では CPU に転送して計算
        _is_rocm = z.is_cuda and torch.version.hip is not None

        if _is_rocm:
            # CPU で iSTFT を実行
            z_cpu = z.cpu()
            win_length = n_fft // (1 + pad)
            x = torch.istft(
                z_cpu,
                n_fft,
                hop_length,
                window=torch.hann_window(win_length).to(z_cpu),
                win_length=win_length,
                normalized=True,
                length=length,
                center=True,
            )
            _, length = x.shape
            result = x.view(*other, length)
            return result
        else:
            # 非 ROCm 環境：元の関数をそのまま使用
            return _orig_ispectro(z, hop_length, length, pad)

    _spec.spectro = _rocm_spectro
    _spec.ispectro = _rocm_ispectro
    _spec._rocm_stft_patched = True


def na1d_rocm(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kernel_size: int,
    dilation: int,
    scale: float,
    rpb: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """純 PyTorch 1D 近傍注意（RPB サポート付き）。

    NATTEN 0.21.x は flex-fna バックエンドを使用するが、以下の制約がある:
      - Triton 不要環境（Windows ROCm）では create_block_mask がフルマスクを実体化して OOM
      - RPB（相対位置バイアス）を加算する API が廃止された
    本関数はこれらを純 PyTorch ループで解決する。

    入出力レイアウト：[B, T, heads, head_dim]（NATTEN heads-last 形式）。
    head_dim の pow2 制約もないためパディング不要。

    境界処理：NATTEN と同一の dilation グループ方式を採用。
      - 各トークン t は同じ dilation グループ (t % dilation) 内のみに注意を向ける。
      - グループ内での位置 gp = t // dilation をもとにウィンドウ中心を計算。
      - 隣接位置の絶対座標：(wc + offset) * dilation + g
      - RPB インデックス：実際の相対グループ位置 + KS - 1

    計算量 O(B·H·T·KS·head_dim)、メモリ O(B·H·T·head_dim) — T²行列を生成しない。

    Args:
        rpb: 相対位置バイアス [H, 2*KS-1] または None（使用しない場合）。
             旧 natten1dqkrpb の RPB と同じテンソルを渡す。
    """
    B, T, H, Dh = q.shape
    half = kernel_size // 2
    GS = T // dilation  # 各 dilation グループの近似サイズ

    q = q.permute(0, 2, 1, 3)  # [B, H, T, Dh]
    k = k.permute(0, 2, 1, 3)
    v = v.permute(0, 2, 1, 3)

    t_idx = torch.arange(T, device=q.device)
    g  = t_idx % dilation        # [T] dilation グループ番号
    gp = t_idx // dilation       # [T] グループ内位置
    wc = gp.clamp(half, max(half, GS - 1 - half))  # [T] ウィンドウ中心（グループ座標）

    scores = []
    v_neighbors = []
    for ki in range(kernel_size):
        offset = ki - half
        # 絶対位置へ変換。端点での安全クランプも付加。
        k_abs = ((wc + offset) * dilation + g).clamp(0, T - 1)  # [T]
        k_sh = k[:, :, k_abs, :]                                  # [B, H, T, Dh]
        score = (q * k_sh).sum(-1) * scale                        # [B, H, T]
        if rpb is not None:
            # 実際の相対グループ位置（境界でウィンドウがスライドした場合も正確に計算）
            rel_pos = (wc + offset) - gp                          # [T]
            rpb_idx = (rel_pos + (kernel_size - 1)).clamp(0, 2 * kernel_size - 2)
            score = score + rpb[:, rpb_idx].unsqueeze(0)          # [B, H, T]
        scores.append(score)
        v_neighbors.append(v[:, :, k_abs, :])                     # [B, H, T, Dh]

    attn = torch.softmax(torch.stack(scores, dim=-1), dim=-1)     # [B, H, T, KS]
    out = sum(attn[..., ki].unsqueeze(-1) * v_neighbors[ki]
              for ki in range(kernel_size))                        # [B, H, T, Dh]
    return out.permute(0, 2, 1, 3)                                # [B, T, H, Dh]


def na2d_rocm(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kernel_size: int,
    dilation: int,
    scale: float,
    rpb: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """純 PyTorch 2D 近傍注意（RPB サポート付き）。

    NATTEN 0.21.x の flex-fna が Triton なしで OOM になる問題と
    RPB 廃止問題を純 PyTorch ループで解決する。

    入出力レイアウト：[B, HH, WW, heads, head_dim]（NATTEN heads-last 形式）。
    境界処理：行・列それぞれ独立に dilation グループ内でウィンドウ中心をクランプ。

    計算量 O(B·H·HH·WW·KS²·head_dim)、メモリ O(B·H·HH·WW·head_dim) — (HH·WW)²行列を生成しない。

    Args:
        rpb: 相対位置バイアス [H, 2*KS-1, 2*KS-1] または None（使用しない場合）。
             旧 natten2dqkrpb の RPB と同じテンソルを渡す。
    """
    B, HH, WW, H, Dh = q.shape
    half = kernel_size // 2
    GS_h = HH // dilation
    GS_w = WW // dilation

    q = q.permute(0, 3, 1, 2, 4)  # [B, H, HH, WW, Dh]
    k = k.permute(0, 3, 1, 2, 4)
    v = v.permute(0, 3, 1, 2, 4)

    hh_idx = torch.arange(HH, device=q.device)
    ww_idx = torch.arange(WW, device=q.device)
    gh  = hh_idx % dilation
    ghp = hh_idx // dilation
    gw  = ww_idx % dilation
    gwp = ww_idx // dilation
    wc_h = ghp.clamp(half, max(half, GS_h - 1 - half))  # [HH]
    wc_w = gwp.clamp(half, max(half, GS_w - 1 - half))  # [WW]

    scores = []
    v_neighbors = []
    for ki in range(kernel_size):
        for kj in range(kernel_size):
            kh = ((wc_h + (ki - half)) * dilation + gh).clamp(0, HH - 1)   # [HH]
            kw = ((wc_w + (kj - half)) * dilation + gw).clamp(0, WW - 1)   # [WW]
            k_sh = k[:, :, kh][:, :, :, kw, :]                              # [B, H, HH, WW, Dh]
            score = (q * k_sh).sum(-1) * scale                              # [B, H, HH, WW]
            if rpb is not None:
                rel_h = (wc_h + (ki - half)) - ghp                          # [HH]
                rel_w = (wc_w + (kj - half)) - gwp                          # [WW]
                rh = (rel_h + (kernel_size - 1)).clamp(0, 2 * kernel_size - 2)
                rw = (rel_w + (kernel_size - 1)).clamp(0, 2 * kernel_size - 2)
                # rpb: [H, 2KS-1, 2KS-1] → rpb[:, rh][:, :, rw]: [H, HH, WW]
                score = score + rpb[:, rh][:, :, rw].unsqueeze(0)           # [B, H, HH, WW]
            scores.append(score)
            v_neighbors.append(v[:, :, kh][:, :, :, kw, :])

    attn = torch.softmax(
        torch.stack(scores, dim=-1), dim=-1
    )  # [B, H, HH, WW, KS²]
    n = kernel_size * kernel_size
    out = sum(attn[..., i].unsqueeze(-1) * v_neighbors[i]
              for i in range(n))                                              # [B, H, HH, WW, Dh]
    return out.permute(0, 2, 3, 1, 4)                                        # [B, HH, WW, H, Dh]


def release_gpu_resources(*tensors) -> None:
    """GPU テンソルを明示的に解放する。

    ROCm では GPU テンソルが参照されたままプロセス終了シーケンスに入ると
    HIP ランタイムがデッドロックしてハングする (CUDAtoROCmPorting.md Bug #12)。
    推論完了後に必ず呼ぶこと。
    """
    for t in tensors:
        del t
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
