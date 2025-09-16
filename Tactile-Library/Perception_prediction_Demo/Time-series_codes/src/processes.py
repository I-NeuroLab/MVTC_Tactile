import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, balanced_accuracy_score
from collections import Counter
import torchaudio

devices = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_confusion_and_balanced_acc(y_true, y_pred):
    conf_mat = confusion_matrix(y_true, y_pred, labels=[0,1,2,3,4])
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    return conf_mat, bal_acc

def rmse(y_pred, y_true):
    y_true = y_true.squeeze().numpy()
    y_pred = y_pred.squeeze().numpy()
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def r2_score_np(y_pred, y_true):
    eps = 1e-8
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / (ss_tot + eps)

AUG_ENABLED = True
AUG_MODE = "replace"
AUG_APPEND_RATIO = 1.0

AUG_CFG = {
    "jitter_prob": 1.0,
    "jitter_sigma_frac": 0.02,

    "tmask_prob": 0.7,
    "tmask_num": 1,
    "tmask_max_frac": 0.08,
    "tmask_value": "mean",

    "mixup_prob": 0.3,
    "mixup_alpha": 0.2,
}

def _apply_jitter(x: torch.Tensor, cfg: dict) -> torch.Tensor:
    """x: (B,C,T). 샘플/채널별 std의 일부만큼 가산 노이즈"""
    if cfg["jitter_sigma_frac"] <= 0.0:
        return x
    B, C, T = x.shape
    # 채널별 per-sample std: (B,C,1)
    std = x.float().std(dim=2, keepdim=True, unbiased=False).clamp(min=1e-6)
    noise = torch.randn_like(x) * (cfg["jitter_sigma_frac"] * std)
    return x + noise

def _apply_time_mask(x: torch.Tensor, cfg: dict) -> torch.Tensor:
    """x: (B,C,T). 샘플당 tmask_num개, 길이~U(1, max_frac*T)"""
    if cfg["tmask_num"] <= 0 or cfg["tmask_max_frac"] <= 0.0:
        return x
    B, C, T = x.shape
    Lmax = max(1, int(cfg["tmask_max_frac"] * T))
    if cfg["tmask_value"] == "mean":
        # 채널별 per-sample mean: (B,C,1)
        fill = x.float().mean(dim=2, keepdim=True)
    else:
        fill = torch.zeros((B, C, 1), dtype=x.dtype, device=x.device)

    x_aug = x.clone()
    for b in range(B):
        for _ in range(cfg["tmask_num"]):
            L = torch.randint(low=1, high=Lmax + 1, size=(1,), device=x.device).item()
            s = torch.randint(low=0, high=max(1, T - L + 1), size=(1,), device=x.device).item()
            x_aug[b, :, s:s+L] = fill[b]
    return x_aug

def _augment_batch(inputs: torch.Tensor, targets: torch.Tensor, cfg: dict):
    """
    증강 파이프라인:
    1) Jitter (가산)  →  2) Time-masking  →  3) Mixup(확률적)
    *gain/scale 민감성을 고려해 '가산' 성격 우선. Mixup은 확률/강도를 낮게 유지.
    """
    x, y = inputs, targets
    # Jitter
    if np.random.rand() < cfg["jitter_prob"]:
        x = _apply_jitter(x, cfg)
    # Time-masking
    if np.random.rand() < cfg["tmask_prob"]:
        x = _apply_time_mask(x, cfg)
    # Mixup
    if np.random.rand() < cfg["mixup_prob"]:
        x, y = _apply_mixup(x, y, cfg)
    return x, y

def _apply_mixup(x: torch.Tensor, y: torch.Tensor, cfg: dict):
    """회귀용 mixup. x: (B,C,T), y: (B,)"""
    if x.size(0) < 2 or cfg["mixup_alpha"] <= 0.0:
        return x, y
    lam = np.random.beta(cfg["mixup_alpha"], cfg["mixup_alpha"])
    perm = torch.randperm(x.size(0), device=x.device)
    x_mix = lam * x + (1.0 - lam) * x[perm]
    y_mix = lam * y + (1.0 - lam) * y[perm]
    return x_mix, y_mix

def train_one_epoch(model, dataloader, optimizer,
                    aug_enabled: bool = None,         # ← 증강 ON/OFF
                    aug_mode: str = None,             # ← "replace"|"append"
                    append_ratio: float = None,       # ← 0.0~1.0 (append일 때만)
                    aug_cfg: dict = None):         # (없으면 전역 AUG_CFG 사용)
    model.train()

    # 1) on/off 결정
    if aug_enabled is None:
        use_aug = AUG_ENABLED
    else:
        use_aug = bool(aug_enabled)

    # 2) 모드/비율 결정
    mode = (AUG_MODE if aug_mode is None else aug_mode)
    mode = mode.lower()
    if mode not in ("replace", "append"):
        mode = "replace"

    ar = AUG_APPEND_RATIO if append_ratio is None else float(append_ratio)
    ar = float(np.clip(ar, 0.0, 1.0))

    # 3) cfg 준비
    if aug_cfg is None:
        aug_cfg = (AUG_CFG if use_aug else None)

    overall_loss = 0.0
    all_preds, all_true = [], []
    first_batch_print = True

    for inputs, targets in dataloader:
        inputs = inputs.to(devices)
        targets = targets.to(devices).float().view(-1)  # (B,)
        
        if use_aug and aug_cfg is not None:
            x_aug, y_aug = _augment_batch(inputs, targets, aug_cfg)

            if first_batch_print:
                if mode == "append":
                    print(f"[AUG ON | mode=append]")
                else:
                    print(f"[AUG ON | mode=replace]")
                first_batch_print = False

            if mode == "append":
                # append_ratio=0 → 사실상 replace와 동일 처리(추가 없음)
                if ar <= 0.0:
                    x_batch, y_batch = x_aug, y_aug
                elif ar >= 1.0:
                    # 전부 추가 → 배치 2배
                    x_batch = torch.cat([inputs, x_aug], dim=0)
                    y_batch = torch.cat([targets, y_aug], dim=0)
                else:
                    # 증강본 중 일부만 추가
                    B = inputs.size(0)
                    k = max(1, int(round(B * ar)))
                    idx = torch.randperm(B, device=inputs.device)[:k]
                    x_batch = torch.cat([inputs, x_aug[idx]], dim=0)
                    y_batch = torch.cat([targets, y_aug[idx]], dim=0)
            else:
                # replace
                x_batch, y_batch = x_aug, y_aug
        else:
            if first_batch_print:
                print("[AUG OFF]")
                first_batch_print = False
            x_batch, y_batch = inputs, targets
        
        optimizer.zero_grad()

        preds = model(inputs)
        preds = preds.view(-1)

        base = F.smooth_l1_loss(preds, targets)
        mx, my = preds.mean(), targets.mean()
        vx, vy = preds.var(unbiased=False), targets.var(unbiased=False)
        cov = ((preds - mx) * (targets - my)).mean()
        ccc = (2*cov) / (vx + vy + (mx - my)**2 + 1e-8)
        loss = base + 0.1 * (1.0 - ccc.clamp(-1, 1))
        loss.backward()
        optimizer.step()

        overall_loss += loss.item()
        all_preds.append(preds.detach().cpu())
        all_true.append(targets.detach().cpu())

    avg_loss = overall_loss / max(1, len(dataloader))
    preds_cat = torch.cat(all_preds).squeeze()
    true_cat  = torch.cat(all_true).squeeze()

    rmse_val = rmse(preds_cat, true_cat)
    r2_val   = r2_score_np(preds_cat, true_cat)

    return avg_loss, preds_cat, rmse_val, true_cat, r2_val

@torch.no_grad()
def validation(model, dataloader):
    model.eval()
    overall_loss = 0.0
    all_preds, all_true = [], []

    for inputs, targets in dataloader:
        inputs = inputs.to(devices)
        targets = targets.to(devices).float().view(-1)  # (B,)

        preds = model(inputs)          # (B,)
        preds = preds.view(-1)

        base = F.smooth_l1_loss(preds, targets)
        mx, my = preds.mean(), targets.mean()
        vx, vy = preds.var(unbiased=False), targets.var(unbiased=False)
        cov = ((preds - mx) * (targets - my)).mean()
        ccc = (2*cov) / (vx + vy + (mx - my)**2 + 1e-8)
        loss = base + 0.1 * (1.0 - ccc.clamp(-1, 1))   # λ=0.05~0.2 탐색
        overall_loss += loss.item()

        all_preds.append(preds.detach().cpu())
        all_true.append(targets.detach().cpu())

    avg_loss = overall_loss / max(1, len(dataloader))
    preds_cat = torch.cat(all_preds).squeeze()
    true_cat  = torch.cat(all_true).squeeze()

    rmse_val = rmse(preds_cat, true_cat)
    r2_val   = r2_score_np(preds_cat, true_cat)

    return avg_loss, preds_cat, rmse_val, true_cat, r2_val
