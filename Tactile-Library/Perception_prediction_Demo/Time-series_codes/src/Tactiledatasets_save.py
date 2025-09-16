# --- Tactiledatasets_save.py : 10x5-fold(Repeated 5-fold) 지원 버전 ---
import numpy as np
import pickle
from scipy import io
import torch
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from torch.utils.data import TensorDataset, DataLoader
import os
from pathlib import Path
from typing import Optional, Union

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
def _paths_for(split_root: Path, rep: int, fold: int):
    rep_dir = split_root / f"rep{rep:02d}"
    rep_dir.mkdir(parents=True, exist_ok=True)
    train_path = rep_dir / f"train_idx_rep{rep:02d}_fold{fold}.pkl"
    val_path   = rep_dir / f"val_idx_rep{rep:02d}_fold{fold}.pkl"
    test_path  = rep_dir / f"test_idx_rep{rep:02d}_fold{fold}.pkl"
    return rep_dir, train_path, val_path, test_path

# ------------------------------
# 메인 함수
# ------------------------------
def TacDataset_save(
    Dir,
    save_dir,
    random_seed: int = 42,
    type_idx: int = 0,
    zscore_mode: str = "per_channel_global",
    shuffle_arrays: bool = False,
    save_stats: bool = True,
    n_repeats: int = 1,
    repeat_idx: int = 0,
    outer_splits: int = 5,
    inner_splits: int = 4,
    splits_root: Optional[Union[str, Path]] = None,
):
    """
    Repeated (n_repeats) × 5-fold (outer_splits) with object-wise stratification.
    - Saves indices to: {splits_root}/repXX/train|val|test_idx_repXX_foldY.pkl
    - Returns the 5 folds for the selected `repeat_idx` as tensor tuples.

    Args:
        Dir:              folder containing X_*.mat and Y_*.mat
        save_dir:         base folder; if `splits_root` is None, indices are saved under save_dir/splits
        random_seed:      base seed
        type_idx:         selects dataname
        zscore_mode:      None | "per_channel_global" | "per_feature"
        shuffle_arrays:   optional array-level shuffle before creating tensors (train only)
        save_stats:       save z-score mean/std per fold
        n_repeats:        e.g., 10 → 10×5-fold
        repeat_idx:       which repeat to load into memory (0..n_repeats-1)
        outer_splits:     outer K (default 5)
        inner_splits:     inner K for train/val (default 4; first split is used)
        splits_root:      where to save indices (default: save_dir/splits)

    Returns:
        Dataset: list length = outer_splits
                 each item: (X_train, Y_train, X_val, Y_val, X_test, Y_test) as Float tensors
    """
    # dataname
    data_list = ['manual','manual_L','manual_R',
                 'multi', 'multi_L','multi_R',
                 'crop_2s','crop_4s', 'crop_8s']
    dataname = data_list[type_idx]

    # paths
    save_dir = Path(save_dir)
    default_dir = Path(Dir)
    if splits_root is None:
        split_root = (save_dir / "splits").resolve()
    else:
        split_root = Path(splits_root).resolve()
    os.makedirs(split_root, exist_ok=True)

    file_x_path = str(default_dir / f'X_{dataname}.mat')
    file_y_path = str(default_dir / f'Y_{dataname}.mat')

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
    Y_reg     = Y[:, 1].astype(np.float32).reshape(-1)
    Y_percep  = Y[:, 2].astype(np.float32).reshape(-1)

    # map object IDs to consecutive [0..n_objects-1] to be safe
    uniq_ids, inv = np.unique(Y_obj_raw, return_inverse=True)
    n_objects = len(uniq_ids)

    # object-wise stacks
    obj_list   = [X[inv == i] for i in range(n_objects)]
    reg_list   = [Y_reg[inv == i] for i in range(n_objects)]
    label_list = [Y_percep[inv == i] for i in range(n_objects)]
    # one label per object for stratification (use the first entry)
    Y_label = np.array([np.ravel(label_list[i])[0] for i in range(n_objects)])

    # ------------------------------
    # Generate/save all repeats if missing
    # ------------------------------
    for rep in range(n_repeats):
        outer_kf = StratifiedKFold(n_splits=outer_splits, shuffle=True, random_state=random_seed + rep)
        for fold, (temp_index, test_index) in enumerate(outer_kf.split(np.arange(n_objects), Y_label)):
            rep_dir, train_path, val_path, test_path = _paths_for(split_root, rep, fold)

            if train_path.exists() and val_path.exists() and test_path.exists():
                continue

            # inner split on remaining temp set
            temp_labels = Y_label[temp_index]
            inner_kf = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=(random_seed + 10_000 + rep))
            train_index_rel, val_index_rel = next(inner_kf.split(np.arange(len(temp_index)), temp_labels))

            train_indices = temp_index[train_index_rel]
            val_indices   = temp_index[val_index_rel]

            with open(train_path, "wb") as f: pickle.dump(train_indices, f)
            with open(val_path,   "wb") as f: pickle.dump(val_indices,   f)
            with open(test_path,  "wb") as f: pickle.dump(test_index,    f)

    # ------------------------------
    # Load selected repeat and build tensors
    # ------------------------------
    rng = np.random.RandomState(random_seed + repeat_idx)
    Dataset = [None] * outer_splits

    for fold in range(outer_splits):
        _, train_path, val_path, test_path = _paths_for(split_root, repeat_idx, fold)
        with open(train_path, "rb") as f: train_indices = pickle.load(f)
        with open(val_path,   "rb") as f: val_indices   = pickle.load(f)
        with open(test_path,  "rb") as f: test_indices  = pickle.load(f)

        # stack samples per object
        X_train = np.vstack([obj_list[i] for i in train_indices])
        Y_train = np.hstack([reg_list[i] for i in train_indices])
        X_val   = np.vstack([obj_list[i] for i in val_indices])
        Y_val   = np.hstack([reg_list[i] for i in val_indices])
        X_test  = np.vstack([obj_list[i] for i in test_indices])
        Y_test  = np.hstack([reg_list[i] for i in test_indices])

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
            torch.tensor(X_val,   dtype=torch.float32),
            torch.tensor(Y_val,   dtype=torch.float32),
            torch.tensor(X_test,  dtype=torch.float32),
            torch.tensor(Y_test,  dtype=torch.float32),
        )

    if zscore_mode: print(f"[TacDataset] z-score: {zscore_mode} (train stats)")
    return Dataset
