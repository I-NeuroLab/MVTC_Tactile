
import torch
import numpy as np
from pathlib import Path
import pandas as pd
from scipy import io
from typing import Dict, Tuple, List
from decimal import Decimal, ROUND_HALF_UP


class EarlyStopping:
    def __init__(self, patience=20, delta=0.0):
        self.patience = patience
        self.delta = delta
        self.best_loss = float('inf')
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                

@torch.no_grad()
def compute_mean_attention_over_loader(model, dataloader, device, mode: str):
    """
    데이터로더 전체에 대해 레이어별 평균 attention을 계산하여 (L,S,S) numpy로 반환.
    - 모델은 eval 모드로 둡니다.
    """
    model.eval()
    attn_sum = None
    count = 0
    token_meta_ref = None

    for xb, yb in dataloader:
        xb = xb.to(device, non_blocking=True)
        y, attn_list, token_meta = model(xb, mode=mode, return_attn=True)

        if attn_sum is None:
            L = len(attn_list)
            S = attn_list[0].size(-1)
            attn_sum = [torch.zeros(S, S, device=device, dtype=attn_list[0].dtype) for _ in range(L)]
            token_meta_ref = token_meta

        # 각 레이어 (B,H,S,S) -> (S,S)로 batch/head 합산
        for li, A in enumerate(attn_list):
            attn_sum[li] += A.sum(dim=(0,1))

        count += attn_list[0].size(0) * attn_list[0].size(1)  # B*H

    attn_mean = torch.stack([M / max(count,1) for M in attn_sum], dim=0)  # (L,S,S)
    return attn_mean.cpu().numpy(), token_meta_ref

def attention_rollout(attn_layers: np.ndarray, add_residual: bool = True, eps: float = 1e-8) -> np.ndarray:
    """
    attn_layers: (L, S, S)  [배치/헤드 평균]
    반환: (S, S) 레이어 누적 주의(rollout). 행=쿼리, 열=키.
    """
    L, S, _ = attn_layers.shape
    A = attn_layers.copy()
    if add_residual:
        I = np.eye(S)[None, :, :]
        A = (A + I) / 2.0
    A = A / (A.sum(axis=-1, keepdims=True) + eps)  # 각 레이어 행 정규화
    R = A[0]
    for l in range(1, L):
        R = A[l].dot(R)
    return R  # (S, S)

def _infer_token_splits(attn_meta: dict, S_total: int):
    """
    CLS 없는 현재 모델 설계 기준 토큰 분할 추론.
    """
    if attn_meta.get("hybrid", False) or ("S_time" in attn_meta):
        S_time = int(attn_meta.get("S_time", 0))
        S_branch = max(S_total - S_time, 0)
    elif attn_meta.get("time_tokens", False):
        S_time, S_branch = S_total, 0
    else:
        # branch-only 또는 메타가 빈 경우는 전부 branch로 간주
        S_time, S_branch = 0, S_total
    return S_time, S_branch

def _build_token_rows_no_cls(attn_meta: dict, S_total: int, T: int, stride: int, fs: float):
    """
    CLS 미사용 설계에 맞춘 토큰 표 생성:
      [time(0..S_time-1), branch(S_time..S_total-1)]
    """
    S_time, S_branch = _infer_token_splits(attn_meta, S_total)
    rows = []
    # time 토큰 구간
    for i in range(S_time):
        start = i * stride
        end   = min((i + 1) * stride - 1, T - 1)
        rows.append(dict(token_idx=i, token_type='time', token_sub=i,
                         start_sample=start, end_sample=end,
                         start_sec=start / fs, end_sec=end / fs))
    # branch 토큰(시간 구간 개념 없음)
    base = S_time
    for j in range(S_branch):
        rows.append(dict(token_idx=base + j, token_type='branch', token_sub=j,
                         start_sample=np.nan, end_sample=np.nan,
                         start_sec=np.nan, end_sec=np.nan))
    return rows, S_time, S_branch

def save_per_token_importance(save_dir_path: Path, baseprefix: str,
                              attn_mean_layers: np.ndarray, attn_meta: dict,
                              model_kwargs: dict, T: int):
    """
    (L,S,S) 유지 저장 + 각 토큰(S)의 중요도 2종:
    - imp_rollout: 레이어 누적 주의 R(S,S)의 '열 평균' (모든 쿼리가 그 키를 얼마나 봤는가) ← 멀티-홉 반영
    - imp_clsmean: 레이어 평균 주의 Ā(S,S)의 '열 평균' (간단 기준선)
    ※ CLS 미사용 설계이므로 'CLS→토큰'이 아니라 '전체→토큰' 중심성으로 정의.
    """
    S = attn_mean_layers.shape[1]
    stride = int(model_kwargs.get("pool_stride", 32))
    fs     = float(model_kwargs.get("fs", 1250.0))

    # 1) rollout 기반
    R = attention_rollout(attn_mean_layers, add_residual=True)  # (S,S)
    imp_rollout = R.mean(axis=0)                                 # (S,)
    imp_rollout = np.maximum(imp_rollout, 0.0)
    imp_rollout /= (imp_rollout.sum() + 1e-8)

    # 2) 레이어 평균 주의의 열 평균(간단 기준선)
    A_bar = attn_mean_layers.mean(axis=0)        # (S,S)
    imp_clsmean = A_bar.mean(axis=0)             # (S,)
    imp_clsmean = np.maximum(imp_clsmean, 0.0)
    imp_clsmean /= (imp_clsmean.sum() + 1e-8)

    # 사람이 읽기 쉬운 표(CLS 없음)
    rows, S_time, S_branch = _build_token_rows_no_cls(attn_meta, S_total=S, T=T, stride=stride, fs=fs)
    assert len(rows) == S, f"토큰 행 개수({len(rows)})와 S({S})가 다릅니다."

    df = pd.DataFrame(rows)
    df["imp_rollout"] = imp_rollout
    df["imp_clsmean"] = imp_clsmean
    df["imp_rollout_norm"] = df["imp_rollout"] / (df["imp_rollout"].sum() + 1e-8)
    df["imp_clsmean_norm"] = df["imp_clsmean"] / (df["imp_clsmean"].sum() + 1e-8)

    # CSV 저장
    df.to_csv(save_dir_path / f"{baseprefix}_TokenImportance.csv", index=False)

    # MAT 저장: (L,S,S) + 벡터
    io.savemat(str(save_dir_path / f"{baseprefix}_TokenImportance.mat"), {
        "attn_mean_layers": attn_mean_layers,  # (L,S,S)
        "imp_rollout": imp_rollout,            # (S,)
        "imp_clsmean": imp_clsmean,            # (S,)
        "S_time": int(S_time),
        "S_branch": int(S_branch),
        "pool_stride": stride,
        "fs": fs,
        "T": int(T),
    })


# numpy 스칼라를 파이썬 기본형으로 캐스팅
def _to_py(x):
    if isinstance(x, np.generic):   # np.float32, np.int32 등
        return x.item()
    return x

_Q = Decimal("0.1")  # 0.1 단위로 양자화

def make_equal_bands(low: float, high: float, n) -> List[Tuple[float, float]]:
    """
    [low, high]를 n등분하여 (start, end) 밴드 목록을 반환.
    모든 경계는 소수 첫째자리까지 반올림되어 0.1 단위로 고정됨.
    n이 numpy.int32 등이어도 동작.
    """
    low = _to_py(low)
    high = _to_py(high)
    n = int(_to_py(n))  # numpy.int32 -> int

    if n <= 0:
        raise ValueError("n must be >= 1")
    if not (low < high):
        raise ValueError("low must be < high")

    L = Decimal(str(low))
    H = Decimal(str(high))
    step = (H - L) / Decimal(n)

    # 경계 생성 + 0.1 단위 양자화(반올림)
    edges = [(L + step * i).quantize(_Q, rounding=ROUND_HALF_UP) for i in range(n + 1)]

    # 혹시 모를 누적 반올림 오차를 시작/끝에서 보정
    edges[0]  = Decimal(str(low)).quantize(_Q, rounding=ROUND_HALF_UP)
    edges[-1] = Decimal(str(high)).quantize(_Q, rounding=ROUND_HALF_UP)

    # (start, end) 밴드로 변환 (float로 반환)
    bands = [(float(edges[i]), float(edges[i + 1])) for i in range(n)]
    return bands


def make_fbank_index(
    fbank_n,
    bounds_per_receptor: Dict[str, Tuple[float, float]],
) -> Dict[str, List[Tuple[float, float]]]:
    """
    receptor별 (low, high)를 받아 n등분 밴드를 생성.
    모든 값은 소수 첫째자리까지만 허용.
    """
    fbank_n = int(_to_py(fbank_n))
    fbank_custom: Dict[str, List[Tuple[float, float]]] = {}

    for name, (low, high) in bounds_per_receptor.items():
        bands = make_equal_bands(low, high, fbank_n)

        # 시작/끝을 한 번 더 확정적으로 보정
        if bands:
            bands[0]  = (float(Decimal(str(low)).quantize(_Q, rounding=ROUND_HALF_UP)), bands[0][1])
            bands[-1] = (bands[-1][0], float(Decimal(str(high)).quantize(_Q, rounding=ROUND_HALF_UP)))

        fbank_custom[name] = bands

    return fbank_custom
