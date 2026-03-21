"""ROCm (AMD GPU / HIP) 固有のバグ回避パッチ。

ドキュメント参照: ROCm対応記録_Allinone用/CUDAtoROCmPorting.md
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
      - LayerNorm GPU fp32 アップキャスト (Bug #1)
      - GroupNorm GPU fp32 アップキャスト (Bug #2)
      - demucs sin_embedding GPU ネイティブ exp 実装 (Bug #3)
      - Conv1d GPU fp32 アップキャスト (Bug #5)
    """
    if not is_rocm():
        return

    # Bug #4: Flash SDP / Mem-Efficient SDP が NaN を生成する
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)

    # Bug #1: LayerNorm 大規模 reduction NaN（GPU fp32 アップキャストで回避）
    _patch_layer_norm()

    # Bug #2: GroupNorm 大規模 reduction NaN（GPU fp32 アップキャストで回避）
    _patch_group_norm()

    # Bug #3: demucs sin_embedding の整数テンソルべき乗が NaN を生成
    _patch_sin_embedding()

    # Bug #5: fp16 Conv1d NaN（GPU fp32 アップキャストで回避）
    _patch_conv1d()


def _patch_layer_norm() -> None:
    """nn.LayerNorm の forward を ROCm NaN セーフ版に差し替える。

    ROCm の GPU カーネルは大規模 reduction で NaN を生成する (CUDAtoROCmPorting.md Bug #1)。
    回避策として入力・weight・bias を fp32 にアップキャストした上で GPU 上の F.layer_norm を呼ぶ。
    reduction が fp32 で行われるため NaN が発生しない。出力は元の dtype に戻す。
    これにより CPU フォールバック版と異なり PCIe 転送が発生せず、推論速度を維持できる。
    """
    if getattr(nn.LayerNorm, '_rocm_patched', False):
        return

    _orig_forward = nn.LayerNorm.forward

    def _rocm_forward(self, x: torch.Tensor) -> torch.Tensor:
        if not (x.is_cuda and torch.version.hip is not None):
            return _orig_forward(self, x)
        orig_dtype = x.dtype
        out = F.layer_norm(
            x.float(),
            self.normalized_shape,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        )
        return out.to(orig_dtype)

    nn.LayerNorm.forward = _rocm_forward
    nn.LayerNorm._rocm_patched = True


def _group_norm_gpu(
    x: torch.Tensor,
    num_groups: int,
    weight: Optional[torch.Tensor],
    bias: Optional[torch.Tensor],
    eps: float,
) -> torch.Tensor:
    """F.group_norm を使わない純 PyTorch GPU 実装。

    ROCm の group_norm カーネル自体が NaN を生成するケースを完全に回避するため、
    基本演算（mean / var / rsqrt）で GroupNorm を再実装する。
    reduction は常に fp32 で行い、出力は入力の dtype に戻す。

    数式:
        x_grouped = x.view(B, G, -1)           # [B, G, N]  N = C//G * spatial
        mean = x_grouped.mean(dim=-1)           # [B, G]
        var  = x_grouped.var(dim=-1, unbiased=False)
        x_norm = (x - mean) / sqrt(var + eps)
        out  = x_norm * weight + bias

    Args:
        x:          入力テンソル [B, C, *]
        num_groups: グループ数 G（C の約数）
        weight:     スケール係数 [C] または None
        bias:       バイアス [C] または None
        eps:        数値安定化定数
    """
    orig_dtype = x.dtype
    xf = x.float()                              # fp32 にアップキャスト

    B, C = xf.shape[0], xf.shape[1]
    spatial_shape = xf.shape[2:]                # () for 1D, (L,) for 2D, ...
    G = num_groups
    N = (C // G) * (xf[0, 0].numel() if spatial_shape else 1)
    # spatial を含めた要素数: C//G × prod(spatial_shape)

    x_grouped = xf.reshape(B, G, -1)           # [B, G, N]  reshape で非連続テンソルも対応

    mean = x_grouped.mean(dim=-1, keepdim=True)                          # [B, G, 1]
    var  = x_grouped.var(dim=-1, keepdim=True, unbiased=False)           # [B, G, 1]
    x_norm = (x_grouped - mean) * (var + eps).rsqrt()                   # [B, G, N]

    x_norm = x_norm.reshape(B, C, *spatial_shape)                       # [B, C, *]

    if weight is not None:
        # weight/bias: [C] → [1, C, 1, 1, ...] でブロードキャスト
        shape = [1, C] + [1] * len(spatial_shape)
        x_norm = x_norm * weight.float().view(shape)
        if bias is not None:
            x_norm = x_norm + bias.float().view(shape)

    return x_norm.to(orig_dtype)


def _patch_group_norm() -> None:
    """nn.GroupNorm の forward を ROCm NaN セーフ版に差し替える。

    F.group_norm を経由せず _group_norm_gpu で GPU 上に直接実装することで
    ROCm の group_norm カーネル自体を完全に回避する。
    """
    if getattr(nn.GroupNorm, '_rocm_patched', False):
        return

    _orig_forward = nn.GroupNorm.forward

    def _rocm_forward(self, x: torch.Tensor) -> torch.Tensor:
        if not (x.is_cuda and torch.version.hip is not None):
            return _orig_forward(self, x)
        return _group_norm_gpu(x, self.num_groups, self.weight, self.bias, self.eps)

    nn.GroupNorm.forward = _rocm_forward
    nn.GroupNorm._rocm_patched = True


def _patch_sin_embedding() -> None:
    """demucs の create_sin_embedding を ROCm GPU ネイティブ実装に差し替える。

    元実装の `max_period ** (adim / (half_dim - 1))` は Python int を底とする
    GPU テンソル累乗であり、ROCm 上で NaN を生成する (CUDAtoROCmPorting.md Bug #3)。

    回避策: `a ** b = exp(b * log(a))` の恒等式を利用して
        max_period ** (adim / (half_dim - 1))
            → exp(adim * (log(max_period) / (half_dim - 1)))
    に書き換える。torch.exp は ROCm GPU で正常動作するため PCIe 転送が不要になる。
    """
    import math

    try:
        import demucs.transformer as _dt
    except ImportError:
        return

    if getattr(_dt, '_rocm_sin_patched', False):
        return

    def _rocm_create_sin_embedding(length, dim, shift=0, device=None, max_period=10000):
        assert dim % 2 == 0
        if device is None:
            if torch.cuda.is_available():
                device = 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'
        _dev = device if isinstance(device, torch.device) else torch.device(str(device))
        pos = shift + torch.arange(length, device=_dev).view(-1, 1, 1)
        half_dim = dim // 2
        adim = torch.arange(half_dim, device=_dev).view(1, 1, -1)
        # ROCm Bug #3 回避: int ** float_tensor の代わりに exp(adim * log(max_period) / ...) を使用。
        # a ** b = exp(b * log(a)) の恒等式により数値的に等価。
        log_max = math.log(float(max_period))
        phase = pos / torch.exp(adim * (log_max / (half_dim - 1)))
        return torch.cat([torch.cos(phase), torch.sin(phase)], dim=-1)

    _dt.create_sin_embedding = _rocm_create_sin_embedding
    _dt._rocm_sin_patched = True


def _patch_conv1d() -> None:
    """nn.Conv1d の forward を ROCm NaN セーフ版に差し替える。

    ROCm の fp16 Conv1d カーネルは数値的バグにより NaN を生成する (CUDAtoROCmPorting.md Bug #5)。
    入力・重み・バイアスを fp32 にアップキャストして F.conv1d を呼ぶことで GPU 上のまま回避する。
    LayerNorm / GroupNorm パッチと同じ戦略で CPU フォールバックによる PCIe 転送を排除する。
    fp16 以外の dtype（fp32, bf16 等）はそのまま元の forward に委譲する。
    """
    if getattr(nn.Conv1d, '_rocm_patched', False):
        return

    _orig_forward = nn.Conv1d.forward

    def _rocm_forward(self, x: torch.Tensor) -> torch.Tensor:
        if not (x.is_cuda and torch.version.hip is not None and x.dtype == torch.float16):
            return _orig_forward(self, x)
        out = F.conv1d(
            x.float(),
            self.weight.float(),
            self.bias.float() if self.bias is not None else None,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
        return out.half()

    nn.Conv1d.forward = _rocm_forward
    nn.Conv1d._rocm_patched = True


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

    入出力レイアウト: [B, T, heads, head_dim]（NATTEN heads-last 形式）。
    head_dim の pow2 制約もないためパディング不要。

    境界処理: NATTEN と同一の dilation グループ方式を採用。
      - 各トークン t は同じ dilation グループ (t % dilation) 内のみに注意を向ける。
      - グループ内での位置 gp = t // dilation をもとにウィンドウ中心を計算。
      - 隣接位置の絶対座標: (wc + offset) * dilation + g
      - RPB インデックス: 実際の相対グループ位置 + KS - 1

    計算量 O(B·H·T·KS·head_dim)、メモリ O(B·H·T·head_dim) — T² 行列を生成しない。

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

    入出力レイアウト: [B, HH, WW, heads, head_dim]（NATTEN heads-last 形式）。
    境界処理: 行・列それぞれ独立に dilation グループ内でウィンドウ中心をクランプ。

    計算量 O(B·H·HH·WW·KS²·head_dim)、メモリ O(B·H·HH·WW·head_dim) — (HH·WW)² 行列を生成しない。

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
