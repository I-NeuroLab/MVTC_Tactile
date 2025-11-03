import math
from typing import Tuple, Optional, Literal, List, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------- FIR 설계 ----------
def _hamming(M: int) -> torch.Tensor:
    n = torch.arange(M, dtype=torch.float32)
    return 0.54 - 0.46 * torch.cos(2 * math.pi * n / (M - 1))

def _sinc(x: torch.Tensor) -> torch.Tensor:
    return torch.where(x == 0,
                       torch.tensor(1.0, dtype=x.dtype, device=x.device),
                       torch.sin(math.pi * x) / (math.pi * x))

def design_lowpass(fc_hz: float, fs: float, M: int) -> torch.Tensor:
    fc = fc_hz / fs
    n = torch.arange(M, dtype=torch.float32)
    m = (M - 1) / 2
    h_ideal = 2 * fc * _sinc(2 * fc * (n - m))
    h = h_ideal * _hamming(M)
    h = h / h.sum()
    return h

def design_bandpass(f1_hz: float, f2_hz: float, fs: float, M: int) -> torch.Tensor:
    if f1_hz >= f2_hz:
        raise ValueError("f1 must be < f2")
    lp2 = design_lowpass(f2_hz, fs, M)
    lp1 = design_lowpass(f1_hz, fs, M)
    h = lp2 - lp1
    h = h / (h.abs().sum() + 1e-8)
    return h

# ---------- 고정형 FIR (depthwise) ----------
class FIRFilter1D(nn.Module):
    def __init__(self, taps_1d: torch.Tensor):
        super().__init__()
        if taps_1d.ndim != 1 or (taps_1d.numel() % 2) == 0:
            raise ValueError("taps must be 1D and odd")
        self.register_buffer("taps", taps_1d.view(1, 1, -1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T = x.shape
        K = self.taps.shape[-1]
        pad = (K - 1) // 2
        w = self.taps.repeat(C, 1, 1)  # (C,1,K)
        return F.conv1d(x, w, padding=pad, groups=C)

# ---------- Conv 블록 ----------
class ConvGNAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int=7, s: int=1, p: Optional[int]=None,
                 gn_groups: int=1, dropout: float=0.0):
        super().__init__()
        if p is None: p = k // 2
        self.conv = nn.Conv1d(in_ch, out_ch, k, s, p, bias=False)
        self.gn   = nn.GroupNorm(gn_groups, out_ch)
        self.act  = nn.GELU()
        self.drop = nn.Dropout(dropout) if dropout>0 else nn.Identity()
    def forward(self, x):
        return self.drop(self.act(self.gn(self.conv(x))))

# ---------- 수용체 브랜치 ----------
class MechanoreceptorBranch(nn.Module):
    def __init__(self, name: str, in_ch: int, fs: float,
                 fir_kind: Literal["lowpass","bandpass"],
                 f1_hz: Optional[float], f2_hz: Optional[float],
                 taps: int, out_ch: int=16, pool_stride: int=8, dropout: float=0.0,
                 fbank: Optional[List[Tuple[float, float]]] = None,
                 share_subband_conv: bool = False):
        
        super().__init__()
        self.name = name
        self.pool = nn.AvgPool1d(kernel_size=pool_stride, stride=pool_stride)
        self._is_fbank = fbank is not None and len(fbank) > 0

        nyq = fs * 0.5

        if not self._is_fbank:
            # --- 기존 단일 필터 경로 (완전 동일) ---
            if fir_kind == "lowpass":
                fc = min(max(1.0, float(f2_hz)), nyq*0.98)
                h  = design_lowpass(fc, fs, taps)
            else:
                f1 = max(0.5, float(f1_hz)); f2 = min(float(f2_hz), nyq*0.98); f1 = min(f1, f2*0.8)
                h  = design_bandpass(f1, f2, fs, taps)
            self.fir   = FIRFilter1D(h)
            self.conv1 = ConvGNAct(in_ch, out_ch, dropout=dropout)
            self.conv2 = ConvGNAct(out_ch, out_ch, dropout=dropout)
        else:
            # --- 필터뱅크 경로 ---
            bands = []
            for (f1, f2) in fbank:
                f1c = max(1.0, float(f1))
                f2c = min(float(f2), nyq*0.98)
                if f1c >= f2c:
                    f1c = max(1.0, min(f2c*0.8, f2c-0.5))
                bands.append((f1c, f2c))

            self.firs = nn.ModuleList([FIRFilter1D(design_bandpass(f1, f2, fs, taps)) for (f1, f2) in bands])

            if share_subband_conv:
                self.conv1 = ConvGNAct(in_ch, out_ch, dropout=dropout)   # 공유
                self.conv2 = ConvGNAct(out_ch, out_ch, dropout=dropout)  # 공유
                self._shared = True
            else:
                self.conv1_list = nn.ModuleList([ConvGNAct(in_ch, out_ch, dropout=dropout) for _ in bands])
                self.conv2_list = nn.ModuleList([ConvGNAct(out_ch, out_ch, dropout=dropout) for _ in bands])
                self._shared = False

            # 서브밴드 concat 후 채널 혼합(원래 out_ch로 복원)
            #self.mix = ConvGNAct(out_ch * len(bands), out_ch, k=1, p=0, dropout=dropout)

    def forward(self, x):  # x: (B, C_in, T)
        if not self._is_fbank:
            z = self.fir(x)
            z = self.conv1(z)
            z = self.conv2(z)
            z = self.pool(z)
            return z  # (B, out_ch, T')
        else:
            zs = []
            for i, fir in enumerate(self.firs):
                zi = fir(x)
                if self._shared:
                    zi = self.conv1(zi)
                    zi = self.conv2(zi)
                else:
                    zi = self.conv1_list[i](zi)
                    zi = self.conv2_list[i](zi)
                zi = self.pool(zi)  # (B, out_ch, T')
                zs.append(zi)
            z_cat = torch.cat(zs, dim=1)          # (B, out_ch*nb, T')
            return z_cat


# ---------- Sine Positional Encoding ----------
class SinePositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int=100000):
        super().__init__()
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0)/d_model))
        pe[:,0::2] = torch.sin(pos*div); pe[:,1::2] = torch.cos(pos*div)
        self.register_buffer("pe", pe)  # (L, D)
    def forward(self, x):               # x: (B, S, D)
        S = x.size(1)
        return x + self.pe[:S].unsqueeze(0)

# ---------- 어텐션 반환형 Transformer ----------
class MHAEncoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_ff: int=256, dropout: float=0.1, norm_first: bool=True):
        super().__init__()
        self.mha = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ff1 = nn.Linear(d_model, dim_ff); self.ff2 = nn.Linear(dim_ff, d_model)
        self.do  = nn.Dropout(dropout); self.act = nn.GELU()
        self.n1  = nn.LayerNorm(d_model); self.n2 = nn.LayerNorm(d_model)
        self.norm_first = norm_first
    def forward(self, x):
        if self.norm_first:
            y = self.n1(x)
            y, attn = self.mha(y, y, y, need_weights=True, average_attn_weights=False)
            x = x + self.do(y)
            y = self.n2(x)
            y = self.ff2(self.do(self.act(self.ff1(y))))
            x = x + self.do(y)
        else:
            y, attn = self.mha(x, x, x, need_weights=True, average_attn_weights=False)
            x = self.n1(x + self.do(y))
            y = self.ff2(self.do(self.act(self.ff1(x))))
            x = self.n2(x + self.do(y))
        return x, attn  # attn: (B, H, S, S)

class MHAEncoder(nn.Module):
    def __init__(self, L: int, d_model: int, nhead: int, dim_ff: int, dropout: float, norm_first: bool=True):
        super().__init__()
        self.layers = nn.ModuleList([MHAEncoderLayer(d_model, nhead, dim_ff, dropout, norm_first) for _ in range(L)])
    def forward(self, x, return_attn: bool=False):
        attn_list = [] if return_attn else None
        for layer in self.layers:
            x, attn = layer(x)
            if return_attn: attn_list.append(attn)
        return (x, attn_list) if return_attn else (x, None)
    
class TactileRoughnessHybridNet(nn.Module):
    def __init__(self,
        in_ch: int=1, fs: float=1250.0, taps: int=129,
        branch_out_ch: int=16, pool_stride: int=8, K: int=1,
        d_model: int=128, nhead: int=4, num_layers: int=2, dim_feedforward: int=256,
        dropout: float=0.1, norm_first: bool=True,
        output_activation: Literal["identity","scaled_sigmoid","scaled_tanh"]="identity",
        output_minmax: Tuple[float,float]=(0.0,4.0),
        mode: Literal["time","branch","hybrid"]="time",
        n_class: int = 1,
        # --- 추가 ---
        fbank_n: int = 3,                      # 각 브랜치를 몇 개 밴드로 쪼갤지 (기본 3)
        fbank_custom: Optional[Dict[str, List[Tuple[float,float]]]] = None,  # 브랜치별 커스텀 밴드
        share_subband_conv: bool = False,      # True면 서브밴드 간 Conv 공유(파라미터 절약)
    ):
        super().__init__()
        self.mode = mode
        self.K = int(K)
        self.output_activation = output_activation
        self.output_min, self.output_max = output_minmax

        if fbank_custom is None:
            fbank_custom = {}
        sa1_default = [(2.0, 8.0), (8.0, 16.0), (16.0, 32.0)]
        sa2_default = [(0.5, 1.5), (1.5, 3.5), (3.5, 8.0)]
        ra1_default = [(8.0, 16.0), (16.0, 32.0), (32.0, 64.0)]
        ra2_default = [(64.0, 128.0), (128.0, 256.0), (256.0, 400.0)]
        
        sa1_fb = fbank_custom.get("SA1", sa1_default)
        sa2_fb = fbank_custom.get("SA2", sa2_default)
        ra1_fb = fbank_custom.get("RA1", ra1_default)
        ra2_fb = fbank_custom.get("RA2", ra2_default)
        
        # 4 수용체 브랜치 (출력 채널 수는 여전히 각 브랜치당 out_ch "하나"로 맞춰 줌)
        self.branches = nn.ModuleList([
            MechanoreceptorBranch("SA1", in_ch, fs, "bandpass", 2.0, 32.0,  taps,
                                  out_ch=branch_out_ch, pool_stride=pool_stride, dropout=dropout*0.5,
                                  fbank=sa1_fb, share_subband_conv=share_subband_conv),
            MechanoreceptorBranch("SA2", in_ch, fs, "lowpass",  None, 8.0,   taps,
                                  out_ch=branch_out_ch, pool_stride=pool_stride, dropout=dropout*0.5,
                                  fbank=sa2_fb, share_subband_conv=share_subband_conv),
            MechanoreceptorBranch("RA1", in_ch, fs, "bandpass", 8.0,  64.0,  taps,
                                  out_ch=branch_out_ch, pool_stride=pool_stride, dropout=dropout*0.5,
                                  fbank=ra1_fb, share_subband_conv=share_subband_conv),
            MechanoreceptorBranch("RA2", in_ch, fs, "bandpass", 64.0, 400.0, taps,
                                  out_ch=branch_out_ch, pool_stride=pool_stride, dropout=dropout*0.5,
                                  fbank=ra2_fb, share_subband_conv=share_subband_conv),
        ])

        self.nb_per_branch = []
        for b in self.branches:
            if hasattr(b, "firs"):
                self.nb_per_branch.append(len(b.firs))
            else:
                self.nb_per_branch.append(1)
                
        total_c = 0
        for nb in self.nb_per_branch:
            total_c += (branch_out_ch * nb)
                
        # A모드(시간-토큰) 투영
        self.proj_time = nn.Linear(total_c, d_model, bias=False)
        
        self.proj_branch_list = nn.ModuleList([
            nn.Linear(branch_out_ch * nb, d_model, bias=False) for nb in self.nb_per_branch
        ])

        self.branch_embed  = nn.Parameter(torch.zeros(4, d_model));  nn.init.normal_(self.branch_embed, std=0.02)
        self.segment_embed = nn.Parameter(torch.zeros(self.K if self.K>1 else 1, d_model)); nn.init.normal_(self.segment_embed, std=0.02)

        self.pos_enc = SinePositionalEncoding(d_model)
        self.encoder = MHAEncoder(num_layers, d_model, nhead, dim_feedforward, dropout, norm_first)

        self.head = nn.Sequential(
            nn.Linear(d_model, d_model//2), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model//2, n_class)
        )

    # --------- 유틸 ---------
    @torch.no_grad()
    def set_mode(self, mode: Literal["time","branch","hybrid"]):
        self.mode = mode

    def _apply_output_activation(self, y: torch.Tensor):
        if self.output_activation == "identity": return y
        lo, hi = self.output_min, self.output_max
        if self.output_activation == "scaled_sigmoid": return lo + (hi-lo)*torch.sigmoid(y)
        if self.output_activation == "scaled_tanh":    return (hi-lo)*(torch.tanh(y)*0.5 + 0.5) + lo
        return y

    def _extract_branch_feats(self, x: torch.Tensor) -> List[torch.Tensor]:
        # 각: (B, Cb, T')
        return [b(x) for b in self.branches]

    def _temporal_pool_K(self, z: torch.Tensor) -> torch.Tensor:
        # z: (B, Cb, T') -> (B, K, Cb)
        if self.K == 1:
            s = z.mean(dim=2, keepdim=False).unsqueeze(1)  # (B,1,Cb)
        else:
            s = F.adaptive_avg_pool1d(z, self.K).transpose(1,2)  # (B,K,Cb)
        return s

    # --------- 토큰 생성 ---------
    def _make_time_tokens(self, feats: List[torch.Tensor]) -> torch.Tensor:
        # concat channel -> (B, total_c, T')
        x_cat = torch.cat(feats, dim=1)
        x_seq = x_cat.transpose(1, 2)           # (B, S=T', total_c)
        x_seq = self.proj_time(x_seq)           # (B, S, d_model)
        x_seq = self.pos_enc(x_seq)             # 사인 위치인코딩
        return x_seq

    def _make_branch_tokens(self, feats: List[torch.Tensor]) -> Tuple[torch.Tensor, Dict]:
        tokens = []
        order  = []
        branch_indices = []
        for bi, (name, z) in enumerate([(b.name, feats[i]) for i,b in enumerate(self.branches)]):
            s = self._temporal_pool_K(z)              # (B,K,Cb)
            t = self.proj_branch_list[bi](s)
            t = t + self.branch_embed[bi].view(1,1,-1)
            t = t + self.segment_embed[:s.size(1)].view(1, s.size(1), -1)
            start = len(tokens)*0 + sum(tok.size(1) for tok in tokens)
            tokens.append(t)
            branch_indices.append(list(range(start, start + s.size(1))))
            for k in range(s.size(1)): order.append((name, k))
        x_seq = torch.cat(tokens, dim=1)              # (B, S=4*K, D)
        x_seq = self.pos_enc(x_seq)
        meta  = {
            "S": x_seq.size(1), "K": self.K,
            "branch_indices": branch_indices, "order": order,
            "branch_tokens": True
        }
        meta  = {"S": x_seq.size(1), "K": self.K, "branch_indices": branch_indices, "order": order}
        return x_seq, meta

    # --------- forward ---------
    def forward(self, x: torch.Tensor, mode: Optional[Literal["time","branch","hybrid"]]=None, return_attn: bool=False):
        """
        x: (B, C_in, T)
        return:
          - if return_attn: (y, attn_list, token_meta)
          - else          : y
        """
        mode = self.mode if mode is None else mode
        feats = self._extract_branch_feats(x)            # 4개 (B,Cb,T')
        
        token_meta = {"mode": mode}
        if mode == "time":
            x_seq = self._make_time_tokens(feats)
            token_meta.update({"S": x_seq.size(1), "time_tokens": True})

        elif mode == "branch":
            x_seq, meta_b = self._make_branch_tokens(feats)
            token_meta.update(meta_b)

        elif mode == "hybrid":
            x_time = self._make_time_tokens(feats)      # (B, S_t, D)
            x_br, meta_b = self._make_branch_tokens(feats)  # (B, S_b, D)
            # concat (시간 먼저, 그다음 브랜치)
            S_t = x_time.size(1)
            x_seq = torch.cat([x_time, x_br], dim=1)    # (B, S_t+S_b, D)
            token_meta.update({"S_time": S_t, "S_branch": x_br.size(1), **{k:v for k,v in meta_b.items() if k!="S"}})
            token_meta["S"] = x_seq.size(1)
            token_meta["hybrid"] = True
        else:
            raise ValueError("mode must be 'time'|'branch'|'hybrid'")

        x_enc, attn_list = self.encoder(x_seq, return_attn=return_attn)
        y = self.head(x_enc.mean(dim=1)).squeeze(-1)
        y = self._apply_output_activation(y)
        return (y, attn_list, token_meta) if return_attn else y

# ---------- 브랜치 주의(Attention) 프린트 ----------
@torch.no_grad()
def print_time_attention_stats(attn_list: List[torch.Tensor], token_meta: Dict, prefix: str=""):
    """
    Summarize time token attention (excludes CLS at index 0).
    """
    if not token_meta.get("time_tokens", False) and not token_meta.get("hybrid", False):
        print(prefix + "[WARN] time tokens not found.")
        return
    S_t = token_meta.get("S_time", 0)
    if S_t <= 0:
        print(prefix + "[WARN] S_time = 0.")
        return
    # skip CLS (index 0)
    offset = 1
    for li, A in enumerate(attn_list):
        Amean = A.mean(dim=(0,1))  # (S,S) including CLS
        sub = Amean[offset:offset+S_t, offset:offset+S_t]
        sub = sub / (sub.sum(dim=-1, keepdim=True) + 1e-8)
        diag = sub.diag().mean().item()
        near = sub.diagonal(offset=1).mean().item() if S_t > 1 else 0.0
        global_m = sub.mean().item()
        print(f"{prefix}Layer {li}: time-attn diag:{diag:.3f} near:{near:.3f} global:{global_m:.3f}")

@torch.no_grad()
def print_branch_attention_stats(attn_list: List[torch.Tensor], token_meta: Dict, prefix: str=""):
    """
    Summarize branch token attention (excludes CLS at index 0).
    Assumes S_branch >= 1
    """
    if not token_meta.get("branch_tokens", False) and not token_meta.get("hybrid", False):
        print(prefix + "[WARN] branch tokens not found.")
        return
    S_b = token_meta.get("S_branch", 0)
    if S_b <= 0:
        print(prefix + "[WARN] S_branch = 0.")
        return
    # For our simple implementation S_b==1; still print self-attn value
    offset = 1 + token_meta.get("S_time", 0)  # CLS + time
    for li, A in enumerate(attn_list):
        Amean = A.mean(dim=(0,1))
        sub = Amean[offset:offset+S_b, offset:offset+S_b]
        val = float(sub.mean().item())
        print(f"{prefix}Layer {li}: branch-attn mean:{val:.3f}")

@torch.no_grad()
def reduce_attn_list_mean(attn_list: List[torch.Tensor]) -> torch.Tensor:
    """
    Average across batch and heads per layer.
    Input: list of L tensors of shape (B,H,S,S)
    Return: (L,S,S)
    """
    mats = []
    for A in attn_list:
        mats.append(A.mean(dim=(0,1)))
    return torch.stack(mats, dim=0)

TacNet = TactileRoughnessHybridNet