# --- Tactiledatasets_save.py : 10x5-fold(Repeated 5-fold) 지원 버전 ---
import numpy as np
import pickle
from scipy import io
import torch
from sklearn.model_selection import KFold, StratifiedKFold
import os
from pathlib import Path
from typing import Optional, Union
import warnings

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module=r"sklearn\.model_selection\._split"
)

devices = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------
# Z-score 유틸
# ------------------------------
def _fit_zscore_stats(X, mode="per_channel_global"):
    """
    Train 데이터로만 통계를 계산.
    X: np.ndarray, shape (N,T) 또는 (N,C,T)
    return: mean, std  (브로드캐스팅 가능한 shape로 반환)
    """
    if mode is None:
        return None, None
    if X.ndim == 2:  # (N,T)
        if mode == "per_feature":
            mean = X.mean(axis=0, keepdims=True)     # (1,T)
            std  = X.std(axis=0,  keepdims=True)
        elif mode == "per_channel_global":
            mean = X.mean(keepdims=True)             # (1,1)
            std  = X.std( keepdims=True)
        else:
            raise ValueError(f"Unknown zscore_mode: {mode}")
    elif X.ndim == 3:  # (N,C,T)
        if mode == "per_feature":
            mean = X.mean(axis=0, keepdims=True)     # (1,C,T)
            std  = X.std( axis=0, keepdims=True)
        elif mode == "per_channel_global":
            mean = X.mean(axis=(0,2), keepdims=True) # (1,C,1)
            std  = X.std( axis=(0,2), keepdims=True)
        else:
            raise ValueError(f"Unknown zscore_mode: {mode}")
    elif X.ndim == 1:
        if mode == "per_feature":
            mean = X.mean(axis=0, keepdims=True)
            std  = X.std( axis=0, keepdims=True)
        elif mode == "per_channel_global":
            mean = X.mean(keepdims=True)
            std  = X.std( keepdims=True)
        else:
            raise ValueError(f"Unknown zscore_mode: {mode}")
    else:
        raise ValueError(f"Unsupported X.ndim={X.ndim}")
    std = np.where(std < 1e-8, 1.0, std)  # 분산 0 보호
    return mean.astype(np.float32), std.astype(np.float32)

def _apply_zscore(X, mean, std):
    if mean is None:  # zscore 꺼짐
        return X
    return (X - mean) / (std + 1e-8)

# ------------------------------
# 내부 헬퍼
# ------------------------------
def _paths_for(split_root: Path, rep: int, fold: int, label_idx: int):
    rep_dir = split_root / f"rep{rep:02d}"
    rep_dir.mkdir(parents=True, exist_ok=True)
    train_path = rep_dir / f"train_idx_y{label_idx}_rep{rep:02d}_fold{fold}.pkl"
    val_path   = rep_dir / f"val_idx_y{label_idx}_rep{rep:02d}_fold{fold}.pkl"
    test_path  = rep_dir / f"test_idx_y{label_idx}_rep{rep:02d}_fold{fold}.pkl"
    return rep_dir, train_path, val_path, test_path

def _create_stratification_labels(reg_list, label_idx):
    if label_idx == 15:
        # 복합 label인 경우 stratification 불가능하므로 None 반환
        return None
    if label_idx == 16:
        # 복합 label인 경우 stratification 불가능하므로 None 반환
        return None
    
    # 각 object의 첫 번째 샘플 label을 대표값으로 사용
    obj_labels = np.array([reg_list[i][0] for i in range(len(reg_list))])
    
    # 반올림하여 정수로 변환
    strat_labels = np.round(obj_labels,0).astype(np.int64)
    
    return strat_labels
# ------------------------------
# 메인 함수
# ------------------------------
def TacDataset_save(
    Dir,
    save_dir,
    label_idx: int = 0,
    random_seed: int = 42,
    zscore_mode: str = "per_channel_global",
    shuffle_arrays: bool = False,
    save_stats: bool = True,
    n_repeats: int = 1,
    repeat_idx: int = 0,
    splits_root: Optional[Union[str, Path]] = None,):
    
    # dataname
    # paths
    save_dir = Path(save_dir)
    default_dir = Path(Dir)
    if splits_root is None:
        split_root = (save_dir / "splits").resolve()
    else:
        split_root = Path(splits_root).resolve()
    os.makedirs(split_root, exist_ok=True)

    file_x_path = str(default_dir / f'X_resample.mat')
    file_y_path = str(default_dir / f'Y_fv+material.mat')

    # load
    Dat_X = io.loadmat(file_x_path)
    Dat   = io.loadmat(file_y_path)
    X = Dat_X['X']  # (N,T) or (N,C,T)

    Y = Dat['Y']
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    # expected columns: obj, reg, percep in any order used by user; keep legacy mapping
    # Here assume [obj, reg, percep]:
    Y_obj_raw = Y[:, 0].astype(np.int64).reshape(-1)
    Y_reg     = Y[:, label_idx].astype(np.float32).reshape(-1)
    Y_mat_raw = Y[:, 11].astype(np.int64).reshape(-1)
        
    # map object IDs to consecutive [0..n_objects-1] to be safe
    uniq_ids, inv = np.unique(Y_obj_raw, return_inverse=True)
    n_objects = len(uniq_ids)

    # object-wise stacks
    obj_list   = [X[inv == i] for i in range(n_objects)]
    reg_list   = [Y_reg[inv == i] for i in range(n_objects)]
    mat_list   = [Y_mat_raw[inv == i] for i in range(n_objects)]
    # one label per object for stratification (use the first entry)

    test_index_list = [None] * 10
    test_index_list[0] = [11, 19, 24, 27, 30, 35]
    test_index_list[1] = [7, 11, 26, 31, 35, 45]
    test_index_list[2] = [6, 14, 21, 30, 35, 49]
    test_index_list[3] = [9, 16, 26, 40, 45, 48]
    test_index_list[4] = [7, 21, 26, 29, 36, 44]
    test_index_list[5] = [6, 12, 26, 30, 35, 45]
    test_index_list[6] = [1, 23, 27, 32, 35, 49]
    test_index_list[7] = [6, 20, 27, 30, 43, 48]
    test_index_list[8] = [12, 19, 23, 32, 42, 46]
    test_index_list[9] = [1, 10, 23, 31, 45, 46]

    strat_labels = _create_stratification_labels(reg_list, label_idx)
    use_stratified = strat_labels is not None
    
    if use_stratified:
        print(f"[TacDataset] Using StratifiedKFold based on rounded labels")
        print(f"  Label distribution: {np.bincount(strat_labels)}")
    else:
        print(f"[TacDataset] Using regular KFold (label_idx={label_idx} is multi-dimensional)")

    outer_splits = 5
    n_reps = len(test_index_list)  # 24
    
    for rep in range(n_reps):
        test_indices = np.array(test_index_list[rep], dtype=int)

        all_indices = np.arange(n_objects)
        trainval_candidates = np.setdiff1d(all_indices, test_indices)
        
        if use_stratified:
            cv = StratifiedKFold(
                n_splits=outer_splits,
                shuffle=True,
                random_state=random_seed + rep,
            )
            cv_iter = cv.split(trainval_candidates,
                               strat_labels[trainval_candidates])
        else:
            cv = KFold(
                n_splits=outer_splits,
                shuffle=True,
                random_state=random_seed + rep,
            )
            cv_iter = cv.split(trainval_candidates)

        for fold, (train_rel, val_rel) in enumerate(cv_iter):
            rep_dir, train_path, val_path, test_path = _paths_for(
                split_root, rep, fold, label_idx
            )

            # 이미 만들어져 있으면 스킵
            if train_path.exists() and val_path.exists() and test_path.exists():
                continue

            train_indices = trainval_candidates[train_rel]
            val_indices   = trainval_candidates[val_rel]

            with open(train_path, "wb") as f:
                pickle.dump(train_indices, f)
            with open(val_path, "wb") as f:
                pickle.dump(val_indices, f)
            with open(test_path, "wb") as f:
                pickle.dump(test_indices, f)

    # ------------------------------
    # Load selected repeat and build tensors
    # ------------------------------
    rng = np.random.RandomState(random_seed + repeat_idx)
    Dataset = [None] * outer_splits

    for fold in range(outer_splits):
        _, train_path, val_path, test_path = _paths_for(split_root, repeat_idx, fold, label_idx)
        with open(train_path, "rb") as f: train_indices = pickle.load(f)
        with open(val_path,   "rb") as f: val_indices   = pickle.load(f)
        with open(test_path,  "rb") as f: test_indices  = pickle.load(f)

        # stack samples per object
        X_train = np.vstack([obj_list[i] for i in train_indices])
        X_val   = np.vstack([obj_list[i] for i in val_indices])
        X_test  = np.vstack([obj_list[i] for i in test_indices])
        
        Y_train = np.hstack([reg_list[i] for i in train_indices])
        Y_val   = np.hstack([reg_list[i] for i in val_indices])
        Y_test  = np.hstack([reg_list[i] for i in test_indices])

        Y_mat_train = np.hstack([mat_list[i] for i in train_indices])
        Y_mat_val   = np.hstack([mat_list[i] for i in val_indices])
        Y_mat_test  = np.hstack([mat_list[i] for i in test_indices])
        
        # z-score
        mean, std = _fit_zscore_stats(X_train, mode=zscore_mode)
        X_train = _apply_zscore(X_train, mean, std)
        X_val   = _apply_zscore(X_val,   mean, std)
        X_test  = _apply_zscore(X_test,  mean, std)

        # optional shuffle on arrays (train only)
        if shuffle_arrays:
            perm = rng.permutation(len(X_train))
            X_train, Y_train = X_train[perm], Y_train[perm]

        if save_stats and zscore_mode:
            zpath = (split_root / f"rep{repeat_idx:02d}" / f"zscore_rep{repeat_idx:02d}_fold{fold}_{zscore_mode}.npz")
            np.savez(zpath, mean=mean, std=std, mode=zscore_mode)

        Dataset[fold] = (
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(Y_train, dtype=torch.float32),
            torch.tensor(Y_mat_train, dtype=torch.long),
            torch.tensor(X_val,   dtype=torch.float32),
            torch.tensor(Y_val,   dtype=torch.float32),
            torch.tensor(Y_mat_val, dtype=torch.long),
            torch.tensor(X_test,  dtype=torch.float32),
            torch.tensor(Y_test,  dtype=torch.float32),
            torch.tensor(Y_mat_test, dtype=torch.long),
        )

    if zscore_mode: print(f"[TacDataset] z-score: {zscore_mode} (train stats)")
    return Dataset
