
# Making List Usable GPU Devices and Choosing GPU
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import random
from src import *
from models import *
import csv

import copy
import pandas as pd
from scipy import io
from pathlib import Path

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
    
def main(args):
    modellist = ['TacNet']
    for mdl_idx in range(0,1):
        for type_idx in range(2,3):
            for mode_idx in range(0,1):
                for actv_idx in range(1,2):
                    print("time = {}".format(type_idx))
                    save_dir = (Path(__file__).resolve().parent / "log")
                    save_dir.mkdir(parents=True, exist_ok=True)
                    modelname = modellist[mdl_idx]
                    random.seed(args.seed)
                    np.random.seed(args.seed)
                    torch.manual_seed(args.seed)
                    torch.cuda.manual_seed(args.seed)
                    args.epochs = 1000
                    args.batch_size = 128 #64
                    args.lr = 5e-4
                    nclass = 1
                    args.pool_stride = 32 #32
                    args.K = 1 #1
                    args.use_aug = False
                    
                    if actv_idx == 0:
                        args.output_activation = "identity"
                    elif actv_idx == 1:
                        args.output_activation = "scaled_sigmoid"
                    else:
                        args.output_activation = "scaled_tanh"
                        
                    if mode_idx == 0:
                        args.mode = "time"
                    elif mode_idx == 1:
                        args.mode = "branch"
                    else:
                        args.mode = "hybrid"
                        
                    devices = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    
                    CV_cnt = 0
                    for rep in range(5):
                        Temp_data = TacDataset_save(args.data_dir,
                                                    save_dir,
                                                    args.seed,
                                                    type_idx,
                                                    zscore_mode="per_channel_global",
                                                    shuffle_arrays=False,
                                                    save_stats=True,
                                                    n_repeats=5,
                                                    repeat_idx=rep,
                                                    outer_splits=5,
                                                    inner_splits=4,
                                                    splits_root=save_dir)
                        
                        for temp in zip(Temp_data):
                            history = pd.DataFrame(columns=[
                                'epoch', 'tr_loss', 'val_loss',
                                'tr_acc', 'val_acc',
                                'model_updated',
                                'te_loss', 'te_acc'
                            ])
                            
                            X_train, Y_train, X_val, Y_val, X_test, Y_test = temp[0]
                            if type_idx == 3 or type_idx == 4 or type_idx == 5:
                                X_train = X_train
                                X_val = X_val
                                X_test = X_test
                            else:
                                X_train = X_train.unsqueeze(1)
                                X_val = X_val.unsqueeze(1)
                                X_test = X_test.unsqueeze(1)
                    
                            train_dataset = TensorDataset(X_train, Y_train)
                            val_dataset = TensorDataset(X_val, Y_val)
                            test_dataset = TensorDataset(X_test, Y_test)
                    
                            train_dataloader = DataLoader(
                                train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
                            val_dataloader = DataLoader(
                                val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
                            test_dataloader = DataLoader(
                                test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
                    
                            n_chans = X_train.shape[1]
                            model_kwargs = dict(
                                in_ch=n_chans, fs=1250.0, taps=129,
                                branch_out_ch=8, pool_stride=args.pool_stride, K=args.K,
                                d_model=128, nhead=4, num_layers=2, dim_feedforward=256,
                                dropout=0.3, mode=args.mode,
                                output_activation=args.output_activation, output_minmax=(0.0, 4.0), n_class=1
                            )
                            model = TactileRoughnessHybridNet(**model_kwargs).to(devices)
            
                            optimizer = torch.optim.AdamW(
                                model.parameters(), lr=args.lr, weight_decay=5e-4)
                            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                                optimizer, T_max=args.epochs)
                    
                            # Track best results for this fold
                            best_loss = float('inf')
                            best_model_state = None
                            
                            # Training loop for this fold
                            early_stopper = EarlyStopping(patience=30, delta=0.001)
                            for epoch in range(args.epochs):
                                tr_loss, _, tr_acc, _, tr_R = train_one_epoch(
                                    model, train_dataloader, optimizer,
                                    aug_enabled=args.use_aug, aug_mode="append", append_ratio=0.5)
                                val_loss, _, val_acc, _, val_R = validation(
                                    model, val_dataloader)
                                torch.cuda.empty_cache()
                    
                                print("\tEpoch_tr", epoch + 1, f"\tAverage Loss: {tr_loss:.4f}",
                                      f"\ttr_rmse: {tr_acc:.4f}", f"\ttr_R^2: {tr_R:.4f}")
                                
                                print("\tEpoch_val", epoch + 1, f"\tAverage Loss: {val_loss:.4f}",
                                      f"\tval_rmse: {val_acc:.4f}", f"\tval_R^2: {val_R:.4f}")
                                
                                scheduler.step()
                    
                                model_updated = False
                                if best_loss > val_loss:
                                    best_loss = val_loss
                                    best_model_state = copy.deepcopy(model.state_dict())
                                    model_updated = True
                                    print("Model updated")
                    
                                history.loc[len(history)] = [
                                    epoch + 1,
                                    tr_loss,
                                    val_loss,
                                    tr_acc,
                                    val_acc,
                                    model_updated,
                                    0,
                                    0
                                ]
                    
                                early_stopper(val_loss)
                                if early_stopper.early_stop:
                                    print("Early stopping triggered at epoch", epoch + 1)
                                    break
                    
                            ckpt_path = save_dir / f"Best_{modelname}_{CV_cnt}_{type_idx}_{args.mode}_{args.output_activation}_bat128.pt"
                            torch.save({"state_dict": best_model_state, "model_kwargs": model_kwargs}, ckpt_path)

                            del model
                            model = TactileRoughnessHybridNet(**model_kwargs).to(devices)
                            model.load_state_dict({k: v.to(devices) for k, v in best_model_state.items()})
                                        
                            tr_loss, tr_pred, tr_acc, tr_true, tr_R2 = validation(
                                model, train_dataloader)
                            val_loss, val_pred, val_acc, val_true, val_R2 = validation(
                                model, val_dataloader)
                            te_loss, te_pred, te_acc, te_true, te_R2 = validation(
                                model, test_dataloader)
                            torch.cuda.empty_cache()
                            
                            print( f"\ttr_loss: {tr_loss:.4f}",
                                  f"\ttr_rmse: {tr_acc:.4f}",
                                  f"\ttr_R^2: {tr_R2:.4f}")
                            
                            print( f"\tval_loss: {val_loss:.4f}",
                                  f"\tval_rmse: {val_acc:.4f}",
                                  f"\tval_R^2: {val_R2:.4f}")
                            
                            print( f"\tte_loss: {te_loss:.4f}",
                                  f"\tte_rmse: {te_acc:.4f}",
                                  f"\tte_R^2: {te_R2:.4f}")
                            
                            # Save attention (mean over heads & batches)
                            attn_mean_layers, attn_meta = compute_mean_attention_over_loader(model, test_dataloader, devices, mode=args.mode)
                            attn_path = save_dir / f"Attn_{modelname}_{CV_cnt}_{type_idx}_{args.mode}_{args.output_activation}_bat128.mat"
                            io.savemat(str(attn_path), {
                                "attn_mean_layers": attn_mean_layers,  # (L,S,S)
                                "attn_L": attn_mean_layers.shape[0],
                                "attn_S": attn_mean_layers.shape[1],
                                "mode": args.mode
                            })
                            
                            baseprefix = f"Attn_{modelname}_{CV_cnt}_{type_idx}_{args.mode}_{args.output_activation}_bat128"
                            save_per_token_importance(save_dir, baseprefix, attn_mean_layers, attn_meta, model_kwargs, T=int(X_train.shape[2]))

                            savedict = {
                                'train_loss': tr_loss,
                                'train_pred': tr_pred,
                                'train_true': tr_true,
                                'train_rmse': tr_acc,
                                'train_R': tr_R2,
                                
                                'val_loss': val_loss,
                                'val_pred': val_pred,
                                'val_true': val_true,
                                'val_rmse': val_acc,
                                'val_R': val_R2,
                                
                                'test_loss': te_loss,
                                'test_pred': te_pred,
                                'test_true': te_true,
                                'test_rmse': te_acc,
                                'test_R': te_R2
                            }
                            
                            history.loc[len(history)] = [
                                epoch + 2,
                                tr_loss,
                                val_loss,
                                tr_acc,
                                val_acc,
                                model_updated,
                                te_loss,
                                te_acc            
                            ]
                            
                            hist_path = save_dir / f'History_{modelname}_{CV_cnt}_{type_idx}_{args.mode}_{args.output_activation}_bat128.csv'
                            reg_path  = save_dir / f'Reg_{modelname}_{CV_cnt}_{type_idx}_{args.mode}_{args.output_activation}_bat128.mat'
                            history.to_csv(hist_path, index=False)
                            io.savemat(str(reg_path), savedict)
                            CV_cnt = CV_cnt+1
                            del history, savedict
                            print("\tSavedir", save_dir)
    
if __name__ == '__main__':

    args = parse_argument()
    subject_results = main(args)
