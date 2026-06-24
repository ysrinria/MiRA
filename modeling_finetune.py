from functools import partial
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import drop_path, to_2tuple, trunc_normal_
from timm.models.registry import register_model
import torch.utils.checkpoint as checkpoint
from typing import Optional

        
def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 400, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': (0.5, 0.5, 0.5), 'std': (0.5, 0.5, 0.5),
        **kwargs
    }


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
    
    def extra_repr(self) -> str:
        return 'p={}'.format(self.drop_prob)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        # x = self.drop(x)
        # commit this for the orignal BERT implement 
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(
            self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
            proj_drop=0., attn_head_dim=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # visualization only
        self.save_attn_vis = False
        self.last_attn = None
        self.last_attn_heads = None

    def forward(self, x):
        B, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)
        
        ## original code
        # q = q * self.scale
        # attn = (q @ k.transpose(-2, -1))

        
        # attn = attn.softmax(dim=-1)
        # attn = self.attn_drop(attn)

        # x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        # x = self.proj(x)
        # x = self.proj_drop(x)
        # return x

        if self.save_attn_vis:
            with torch.no_grad():
                logits_vis = torch.matmul(q * self.scale, k.transpose(-2, -1))   # [B,H,L,L]
                attn_vis = logits_vis.softmax(dim=-1)
                self.last_attn_heads = attn_vis.detach()          # [B,H,L,L]
                self.last_attn = attn_vis.mean(dim=1).detach()    # [B,L,L]

        ## flash Attention code for torch>2.2.0 (A100/H100, CUDA>12.0)
        drop_p = self.attn_drop.p if self.training else 0.0
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p = drop_p,
            is_causal=False,
            scale=self.scale,
        )

        x = out.transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


###################### NAIIVE BASELINES #############################
class Attention_with_intraframe(nn.Module):
    """
    input:: x: [B, N, C],  N = T * Np, 
    y = (1-λ)*y_global + λ*y_intra
    """
    def __init__(
        self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.0,
        proj_drop=0., attn_head_dim=None, intra_lambda=0.5,
        tokens_per_frame=None, 
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # visualization only
        self.save_attn_vis = False
        self.last_attn = None           # [B, L, L]
        self.last_attn_heads = None     # [B, H, L, L]

        self.intra_lambda = float(intra_lambda)
        self.tokens_per_frame = tokens_per_frame

        self._cached_mask: Optional[torch.Tensor] = None  # [N,N], bool
        self._cached_mask_meta: Optional[tuple[int, int, torch.device]] = None  # (N, Np, device)

    @staticmethod
    def _build_intra_mask(N: int, Np: int, device, dtype=torch.bool):
        assert Np is not None and Np > 0, "tokens_per_frame (Np) must be set."
        assert N % Np == 0, f"N ({N}) must be a multiple of tokens_per_frame Np ({Np})."
        frame_ids = torch.arange(N, device=device) // Np  # [N]
        same = (frame_ids[:, None] == frame_ids[None, :]) # [N,N]
        attn_mask = ~same  # True=mask-out(No outside the frame), False=allow
        return attn_mask.to(dtype)

    def _get_mask(self, N: int, Np: int, device):
        return self._build_intra_mask(N, Np, device)

    def forward(self, x: torch.Tensor):
        """
        x: [B, N, C],  N=T*Np
        """
        B, N, C = x.shape
        Np = self.tokens_per_frame
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, H, N, Dh]

        drop_p = self.attn_drop.p if self.training else 0.0

        # (1) Global attention
        y_global = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=drop_p,
            is_causal=False,
            scale=self.scale,
        )  # [B,H,N,Dh]

        # (2) intra-frame attention
        intra_mask = self._get_mask(N, Np, x.device)  # [N,N], bool
        y_intra = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=intra_mask,
            dropout_p=drop_p,
            is_causal=False,
            scale=self.scale,
        )  # [B,H,N,Dh]

        # (3) blending outputs
        lam = self.intra_lambda
        y = (1.0 - lam) * y_global + lam * y_intra  

        # (4) projection
        y = y.transpose(1, 2).contiguous().reshape(B, N, -1)  # [B,N,all_head_dim]
        y = self.proj(y)
        y = self.proj_drop(y)
        return y


class ST_Block(nn.Module): # spatial temporal Block

    def __init__(self, dim, num_heads, mlp_ratio=3., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 attn_head_dim=None, num_seq=None, add_intra_attention=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, attn_head_dim=attn_head_dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        # adjust hyper-params of framewise module (the number of params in vit_Block = the number of params in ST_Block) 
        attn_head_dim_fw = max(32, dim//8)  # NOTE: attn_head_dim = dim // 8
        hidden_mlp_fw = max(64, dim//4)         # NOTE: dim = dim // 4

        self.framewise_norm1 = norm_layer(dim)
        self.framewise_attn = Attention(
            dim, num_heads=3, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, attn_head_dim=attn_head_dim_fw)
        self.framewise_drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.framewise_norm2 = norm_layer(dim)
        self.framewise_mlp = Mlp(in_features=dim, hidden_features=hidden_mlp_fw, act_layer=act_layer, drop=drop)

        if init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None
        
        self.num_seq = int(num_seq)

    def forward(self, x):
        # faily feature attention
        if self.gamma_1 is None:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        
        # Framewise feature attention
        B, num_x, D = x.shape
        num_x_per_seq = int(num_x / self.num_seq)
        
        x = x.view(B, self.num_seq, num_x_per_seq, D)
        x = x.reshape(B * self.num_seq, num_x_per_seq, D)

        x = x + self.framewise_drop_path(self.framewise_attn(self.framewise_norm1(x)))
        x = x + self.framewise_drop_path(self.framewise_mlp(self.framewise_norm2(x)))
        
        x = x.view(B, self.num_seq, num_x_per_seq, -1) 
        x = x.reshape(B, self.num_seq * num_x_per_seq, -1)  
        return x


###################### MY CODE #############################
class AttentionFMP(nn.Module):
    """
    Frame-Marginal Micro-dynamics Reweighted Attention (MiRA).

    Two execution modes:
      - Exact mode:
          post-softmax framewise reweighting + renormalization
      - FlashLite mode:
          single-pass SDPA with per-key log-bias

    Input:
        x: [B, L, C], where L = T * N
           T: number of frames
           N: tokens per frame
    """
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.,
        proj_drop=0.,
        attn_head_dim=None,
        num_seq=None,

        # MiRA hyperparameters
        beta=1.5,
        lam=0.1,
        eps=1e-6,
        floor_eps=1e-3,
        w_self=0.5,
        w_entropy=0.5,
        alpha_min=0.7,
        alpha_max=1.4,   # narrow range improves stability
        flash_tau=0.17,  # 0.22 for HUGE

        # EMA for batch-level frame statistics
        use_ema=True,
        ema_m=0.9,

        # Residual mixing strength
        use_residual=True,
        mix_eta=0.5,

        # FlashLite flag
        use_fmp_flashlite=False,

        # Frame-statistics mode
        # - batch_ema: batch mean + EMA (default for pretraining)
        # - batch:     batch mean only
        # - instance:  per-sample statistics (often useful for finetuning)
        stats_mode: str = "batch_ema",

        # Visualization only
        save_attn_vis=False,

        # Diagnostics only (ablations, analysis, etc.)
        save_internal_stats=False,
    ):
        super().__init__()

        self.num_heads = num_heads
        head_dim = attn_head_dim if attn_head_dim is not None else dim // num_heads
        all_head_dim = head_dim * num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # QKV and output projection
        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # Sequence metadata
        assert num_seq is not None and int(num_seq) > 0
        self.num_seq = int(num_seq)  # T

        # MiRA parameters
        self.beta = float(beta)
        self.lam = float(lam)
        self.eps = float(eps)
        self.floor_eps = float(floor_eps)
        self.w_self = float(w_self)
        self.w_entropy = float(w_entropy)
        self.alpha_min = float(alpha_min)
        self.alpha_max = float(alpha_max)
        self.alpha_min, self.alpha_max = min(self.alpha_min, self.alpha_max), max(self.alpha_min, self.alpha_max)
        self.flash_tau = float(flash_tau)

        # EMA buffer for batch-level confidence prior
        self.use_ema = bool(use_ema)
        self.ema_m = float(ema_m)
        self.register_buffer(
            "c_buf",
            torch.full((1, self.num_seq), 1.0 / self.num_seq),
            persistent=True
        )

        # Residual mixing
        self.use_residual = bool(use_residual)
        self.mix_eta = float(mix_eta)

        # FlashLite
        self.use_fmp_flashlite = bool(use_fmp_flashlite)

        # Statistics mode
        self.stats_mode = str(stats_mode).lower().strip()
        valid_modes = {"batch_ema", "batch", "instance"}
        if self.stats_mode not in valid_modes:
            raise ValueError(
                f"Invalid stats_mode '{self.stats_mode}'. "
                f"Supported modes are: {sorted(valid_modes)}."
            )
        if self.stats_mode != "batch_ema":
            self.use_ema = False

        # Visualization buffers
        self.save_attn_vis = bool(save_attn_vis)
        self.last_attn_orig = None       # [B, L, L]
        self.last_attn = None            # [B, L, L]
        self.last_attn_orig_heads = None # [B, H, L, L]
        self.last_attn_heads = None      # [B, H, L, L]
        self.last_attn_reweighted = None
        self.last_attn_reweighted_heads = None

        # Diagnostics buffers
        self.save_internal_stats = bool(save_internal_stats)

        # internal statistics
        self.last_c = None
        self.last_H = None
        self.last_f = None
        self.last_pi = None
        self.last_alpha = None

    def _sync_cbuf(self, ref: torch.Tensor):
        """Keep the EMA buffer on the same dtype/device as the current statistics."""
        if (self.c_buf.dtype != ref.dtype) or (self.c_buf.device != ref.device):
            self.c_buf.data = self.c_buf.data.to(dtype=ref.dtype, device=ref.device)
    
    def _compute_frame_statistics(self, c_src, H_src, normalize_mode="exact"):
        f = self.w_self * c_src + self.w_entropy * H_src

        if normalize_mode == "exact":
            f = (f - f.min(dim=1, keepdim=True).values) / (
                f.max(dim=1, keepdim=True).values
                - f.min(dim=1, keepdim=True).values
                + 1e-12
            )
            pi_hat = (f + self.eps) ** self.beta
            pi_hat = pi_hat / pi_hat.sum(dim=1, keepdim=True).clamp_min(1e-12)

        elif normalize_mode == "flash":
            f = (f - f.mean(dim=1, keepdim=True)) / (
                f.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-6)
            )

            pi_hat = torch.softmax(self.flash_tau * f, dim=1)

        else:
            raise ValueError(normalize_mode)

        pi = (1.0 - self.lam) * pi_hat + self.lam * (1.0 / self.num_seq)
        return f, pi

    @torch.no_grad()
    def _alpha_from_key_energy(self, k: torch.Tensor) -> torch.Tensor:
        """
        Build framewise scaling factors from key-energy statistics.

        Used in FlashLite mode only.

        Args:
            k: key tensor [B, H, L, Dh]

        Returns:
            alpha:
                [B, T] in instance mode
                [1, T] in batch / batch_ema mode
        """
        B, H, L, Dh = k.shape
        T = self.num_seq
        assert L % T == 0, f"[MiRA] L({L}) must be divisible by T({T})."
        N = L // T

        # Per-token key energy
        key_pow = (k ** 2).mean(dim=(1, 3))   # [B, L]
        KTN = key_pow.reshape(B, T, N)        # [B, T, N]

        # Framewise confidence proxy
        c_hat = KTN.sum(dim=-1)                                       # [B, T]
        c_norm = c_hat / c_hat.sum(-1, keepdim=True).clamp_min(1e-12)

        # Framewise inverse entropy proxy
        p_hat = KTN / KTN.sum(-1, keepdim=True).clamp_min(1e-12)
        H_raw = -(p_hat.clamp_min(1e-12) * p_hat.clamp_min(1e-12).log()).sum(-1)
        H = 1.0 / (H_raw + 1e-6)

        if self.stats_mode == "instance":
            c_src = c_norm
            H_src = H

        elif self.stats_mode in ("batch", "batch_ema"):
            c_mean = c_norm.mean(dim=0, keepdim=True)  # [1, T]
            H_mean = H.mean(dim=0, keepdim=True)       # [1, T]

            self._sync_cbuf(c_mean)

            if self.stats_mode == "batch_ema" and self.use_ema and self.training:
                self.c_buf.mul_(self.ema_m).add_((1.0 - self.ema_m) * c_mean)
                c_src = self.c_buf
                H_src = H_mean
            else:
                c_src = c_mean
                H_src = H_mean

        else:
            raise ValueError(f"Invalid stats_mode: {self.stats_mode}")

        f, pi = self._compute_frame_statistics(c_src, H_src, normalize_mode="flash")
        alpha = (pi * self.num_seq).clamp(self.alpha_min, self.alpha_max)
        return alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, C = x.shape
        T = self.num_seq
        assert L % T == 0, f"[MiRA] L({L}) must be divisible by T({T})."
        N = L // T

        # QKV projection
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((
                self.q_bias,
                torch.zeros_like(self.v_bias, requires_grad=False),
                self.v_bias
            ))

        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, L, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, H, L, Dh]

        # ---------------------------------------------------------
        # FlashLite mode: one-pass SDPA with per-key log-bias
        # ---------------------------------------------------------
        if self.use_fmp_flashlite:
            with torch.no_grad():
                alpha = self._alpha_from_key_energy(k.detach())  # [B,T] or [1,T]

                # ===== diagnostics only =====
                if self.save_internal_stats:
                    self.last_alpha = alpha.detach()

                    Bk, Hk, Lk, Dhk = k.shape
                    Tk = self.num_seq
                    Nk = Lk // Tk
                    key_pow = (k ** 2).mean(dim=(1, 3))   # [B, L]
                    KTN = key_pow.reshape(Bk, Tk, Nk)
                    c_hat = KTN.sum(dim=-1)
                    c_norm = c_hat / c_hat.sum(-1, keepdim=True).clamp_min(1e-12)
                    p_hat = KTN / KTN.sum(-1, keepdim=True).clamp_min(1e-12)
                    H_raw = -(p_hat.clamp_min(1e-12) * p_hat.clamp_min(1e-12).log()).sum(-1)
                    H = 1.0 / (H_raw + 1e-6)
                    if self.stats_mode == "instance":
                        c_src = c_norm
                        H_src = H
                    else:
                        c_src = c_norm.mean(dim=0, keepdim=True)
                        H_src = H.mean(dim=0, keepdim=True)
                    f, pi = self._compute_frame_statistics(c_src, H_src, normalize_mode="flash")
                    self.last_c = c_src.detach()
                    self.last_H = H_src.detach()
                    self.last_f = f.detach()
                    self.last_pi = pi.detach()

                if alpha.shape[0] == 1 and B > 1:
                    alpha = alpha.expand(B, -1)
                log_alpha = alpha.clamp_min(1e-12).log()

            # Match the exact-mode semantics:
            # FlashLite applies a single-pass additive logit bias.
            eta = 1.0
            b_vec = (eta * log_alpha.detach()).repeat_interleave(N, dim=1)
            b_vec = b_vec.to(dtype=q.dtype, device=q.device)

            # Q/K/V augmentation trick:
            # logits = (q @ k^T) * scale + b_k
            B_, H_, L_, Dh_ = q.shape
            q_extra = torch.full((B_, H_, L_, 1), 1.0 / self.scale, device=q.device, dtype=q.dtype)
            k_extra = b_vec[:, None, :, None].expand(B_, H_, L_, 1)
            v_extra = torch.zeros((B_, H_, L_, 1), device=v.device, dtype=v.dtype)

            q_aug = torch.cat([q, q_extra], dim=-1)
            k_aug = torch.cat([k, k_extra], dim=-1)
            v_aug = torch.cat([v, v_extra], dim=-1)

            # Save attention maps only for analysis
            if self.save_attn_vis:
                with torch.no_grad():
                    logits_orig = torch.matmul(q * self.scale, k.transpose(-2, -1))
                    attn_orig_vis = logits_orig.softmax(dim=-1)
                    self.last_attn_orig_heads = attn_orig_vis.detach()
                    self.last_attn_orig = attn_orig_vis.mean(dim=1).detach()

                    logits_vis = logits_orig + b_vec[:, None, None, :]
                    attn_vis = logits_vis.softmax(dim=-1)
                    self.last_attn_heads = attn_vis.detach()
                    self.last_attn = attn_vis.mean(dim=1).detach()

            drop_p = self.attn_drop.p if self.training else 0.0

            with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=True, enable_mem_efficient=True):
                out_aug = F.scaled_dot_product_attention(
                    q_aug,
                    k_aug,
                    v_aug,
                    attn_mask=None,
                    dropout_p=drop_p,
                    is_causal=False,
                    scale=self.scale,
                )

            out = out_aug[..., :Dh_]  # drop the auxiliary dimension
            y = out.transpose(1, 2).reshape(B, L, -1)

        # ---------------------------------------------------------
        # Exact mode: post-softmax reweight + renormalization
        # ---------------------------------------------------------
        else:
            q = q * self.scale
            logits = q @ k.transpose(-2, -1)   # [B, H, L, L]
            A_soft = logits.softmax(dim=-1)
            A_orig = A_soft

            # Framewise confidence
            A_h = A_soft.mean(dim=1)                          # [B, L, L]
            A_h_key = A_h.reshape(B, L, T, N)
            c = A_h_key.sum(-1).mean(1).clamp_min(1e-6)       # [B, T]
            c_norm = c / c.sum(-1, keepdim=True).clamp_min(1e-12)

            # Framewise inverse entropy
            p_keys = A_h.sum(dim=1)                           # [B, L]
            p_frame = p_keys.reshape(B, T, N)
            p_norm = p_frame / p_frame.sum(-1, keepdim=True).clamp_min(1e-12)
            H_raw = -(p_norm.clamp_min(1e-12) * p_norm.clamp_min(1e-12).log()).sum(-1)
            H = 1.0 / (H_raw + 1e-6)

            if self.stats_mode == "instance":
                c_src = c_norm
                H_src = H

            elif self.stats_mode in ("batch", "batch_ema"):
                c_mean = c_norm.mean(dim=0, keepdim=True)  # [1, T]
                H_mean = H.mean(dim=0, keepdim=True)       # [1, T]

                self._sync_cbuf(c_mean)

                if self.stats_mode == "batch_ema" and self.use_ema and self.training:
                    with torch.no_grad():
                        self.c_buf.mul_(self.ema_m).add_((1.0 - self.ema_m) * c_mean)
                    c_src = self.c_buf
                    H_src = H_mean
                else:
                    c_src = c_mean
                    H_src = H_mean

            else:
                raise ValueError(f"Invalid stats_mode: {self.stats_mode}")

            f, pi = self._compute_frame_statistics(c_src, H_src, normalize_mode="exact")

            # Framewise scaling factor
            c_det = c_norm.detach().clamp_min(1e-5)
            alpha = (pi / c_det).clamp(self.alpha_min, self.alpha_max)

            if self.save_internal_stats:
                self.last_c = c_src.detach()
                self.last_H = H_src.detach()
                self.last_f = f.detach()
                self.last_pi = pi.detach()
                self.last_alpha = alpha.detach()

            # Reweight and renormalize
            A4 = A_orig.reshape(B, self.num_heads, L, T, N) * alpha[:, None, None, :, None]
            if self.floor_eps > 0:
                U = 1.0 / (T * N)
                A4 = (1.0 - self.floor_eps) * A4 + self.floor_eps * U

            denom = A4.sum(dim=(3, 4), keepdim=True).clamp_min(1e-12)
            A_mira = (A4 / denom).reshape(B, self.num_heads, L, L)

            # Residual mixing without gate
            if self.use_residual:
                attn = A_orig + self.mix_eta * (A_mira - A_orig)
            else:
                attn = A_mira
            attn = self.attn_drop(attn)

            if self.save_attn_vis:
                # before MiRA
                self.last_attn_orig_heads = A_soft.detach()
                self.last_attn_orig = A_soft.mean(dim=1).detach()
                # pure MiRA reweighted attention, before residual mixing
                self.last_attn_reweighted_heads = A_mira.detach()
                self.last_attn_reweighted = A_mira.mean(dim=1).detach()
                # final attention actually used for forward
                self.last_attn_heads = attn.detach()
                self.last_attn = attn.mean(dim=1).detach()

            y = (attn @ v).transpose(1, 2).reshape(B, L, -1)

        # Output projection
        y = self.proj(y)
        y = self.proj_drop(y)

        return y


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 attn_head_dim=None, num_seq=None, add_intra_attention=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        if add_intra_attention:
            self.attn = Attention_with_intraframe(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, attn_head_dim=attn_head_dim, tokens_per_frame=num_seq)
        else:
            self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, attn_head_dim=attn_head_dim)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x):
        if self.gamma_1 is None:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, num_frames=16, tubelet_size=2):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.tubelet_size = int(tubelet_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0]) * (num_frames // self.tubelet_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv3d(in_channels=in_chans, out_channels=embed_dim, 
                            kernel_size = (self.tubelet_size,  patch_size[0],patch_size[1]), 
                            stride=(self.tubelet_size,  patch_size[0],  patch_size[1]))
        self.num_frames = num_frames

    def forward(self, x, **kwargs):
        B, C, T, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x
    
# sin-cos position encoding
# https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/transformer/Models.py#L31
def get_sinusoid_encoding_table(n_position, d_hid): 
    ''' Sinusoid position encoding table ''' 
    # TODO: make it with torch instead of numpy 
    def get_position_angle_vec(position): 
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)] 

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)]) 
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2]) # dim 2i 
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2]) # dim 2i+1 

    return  torch.tensor(sinusoid_table,dtype=torch.float, requires_grad=False).unsqueeze(0) 


class VisionTransformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, 
                 img_size=224, 
                 patch_size=16, 
                 in_chans=3, 
                 num_classes=1000, 
                 embed_dim=768, 
                 depth=12,
                 num_heads=12, 
                 mlp_ratio=4., 
                 qkv_bias=False, 
                 qk_scale=None, 
                 fc_drop_rate=0., 
                 drop_rate=0., 
                 attn_drop_rate=0.,
                 drop_path_rate=0., 
                 norm_layer=nn.LayerNorm, 
                 init_values=0.,
                 use_learnable_pos_emb=False, 
                 init_scale=0.,
                 all_frames=16,
                 tubelet_size=2,
                 use_checkpoint=False,
                 use_mean_pooling=True,
                 use_st_block=False,
                 add_intra_attention=False,
                 add_fmp_attention=False,
                 fmp_num_last_layers=1,
                 fmp_no_use_ema=False,
                 fmp_use_residual=False,
                 use_fmp_flashlite=False,
                 flash_tau=0.17,
                 stats_mode="batch_ema"):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.tubelet_size = tubelet_size
        self.num_heads = num_heads
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, num_frames=all_frames, tubelet_size=self.tubelet_size)
        num_patches = self.patch_embed.num_patches
        self.use_checkpoint = use_checkpoint
        self.use_st_block = use_st_block
        self.add_intra_attention = add_intra_attention
        self.add_fmp_attention = add_fmp_attention
        self.use_fmp_flashlite = use_fmp_flashlite
        self.stats_mode = stats_mode
        self.flash_tau = float(flash_tau)

        if use_learnable_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        else:
            # sine-cosine positional embeddings is on the way
            self.pos_embed = get_sinusoid_encoding_table(num_patches, embed_dim)
        
        num_seq = int(self.patch_embed.num_frames / self.patch_embed.tubelet_size)
        self.num_seq  = num_seq
        if self.use_st_block:
            Block_cls = ST_Block
        else:
            Block_cls = Block

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block_cls(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values, num_seq=num_seq, add_intra_attention=self.add_intra_attention)
            for i in range(depth)])
        self.norm = nn.Identity() if use_mean_pooling else norm_layer(embed_dim)
        self.fc_norm = norm_layer(embed_dim) if use_mean_pooling else None
        self.fc_dropout = nn.Dropout(p=fc_drop_rate) if fc_drop_rate > 0 else nn.Identity()
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        # === Frame Marginal Projection ===
        if self.add_fmp_attention:        
            self.fmp_use_ema = not(fmp_no_use_ema)
            self.fmp_use_residual = fmp_use_residual    
            self._apply_fmp(num_patches, num_seq, depth, fmp_num_last_layers)

        if use_learnable_pos_emb:
            trunc_normal_(self.pos_embed, std=.02)

        trunc_normal_(self.head.weight, std=.02)
        self.apply(self._init_weights)

        self.head.weight.data.mul_(init_scale)
        self.head.bias.data.mul_(init_scale)

    def _apply_fmp(self, num_total_tokens, num_seq, num_block_depth, fmp_num_last_layers):
        assert num_seq > 0 and num_total_tokens % num_seq == 0, \
            f"num_total_tokens={num_total_tokens}, T={num_seq} => It must be L=T*N ."
        
        start = max(0, num_block_depth - int(fmp_num_last_layers))
        for i in range(start, num_block_depth):
            attn_origin = self.blocks[i].attn
            num_heads_i = getattr(attn_origin, "num_heads", getattr(self, "num_heads", None))
            qkv_bias_i = (getattr(attn_origin, "q_bias", None) is not None)
            qk_scale_i = getattr(attn_origin, "scale", None)
            attn_drop_i = getattr(attn_origin, "attn_drop", nn.Dropout(0.0)).p if hasattr(attn_origin, "attn_drop") else 0.0
            proj_drop_i = getattr(attn_origin, "proj_drop", nn.Dropout(0.0)).p if hasattr(attn_origin, "proj_drop") else 0.0
            self.blocks[i].attn = AttentionFMP(
                dim=self.embed_dim, num_heads=num_heads_i,
                qkv_bias=qkv_bias_i, qk_scale=qk_scale_i,
                attn_drop=attn_drop_i, proj_drop=proj_drop_i,
                num_seq=num_seq,
                use_ema=bool(self.fmp_use_ema),
                use_residual=bool(self.fmp_use_residual),
                use_fmp_flashlite=bool(self.use_fmp_flashlite),
                flash_tau=self.flash_tau,
                stats_mode=self.stats_mode,
                save_attn_vis=False,
                save_internal_stats=False,
            )

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        x = self.patch_embed(x)
        B, _, _ = x.size()

        if self.pos_embed is not None:
            x = x + self.pos_embed.expand(B, -1, -1).type_as(x).to(x.device).clone().detach()
        x = self.pos_drop(x)

        if self.use_checkpoint:
            for blk in self.blocks:
                x = checkpoint.checkpoint(blk, x)
        else:   
            for blk in self.blocks:
                x = blk(x)

        x = self.norm(x)
        if self.fc_norm is not None:
            return self.fc_norm(x.mean(1))
        else:
            return x[:, 0]

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(self.fc_dropout(x))
        return x


@register_model
def vit_small_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def vit_base_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def vit_base_patch16_384(pretrained=False, **kwargs):
    model = VisionTransformer(
        img_size=384, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def vit_large_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def vit_large_patch16_384(pretrained=False, **kwargs):
    model = VisionTransformer(
        img_size=384, patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def vit_large_patch16_512(pretrained=False, **kwargs):
    model = VisionTransformer(
        img_size=512, patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def vit_huge_patch16_224(pretrained=False, **kwargs):
    # kwargs["flash_tau"] = 0.22  # to change the default tau for huge model
    model = VisionTransformer(
        patch_size=16, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model






