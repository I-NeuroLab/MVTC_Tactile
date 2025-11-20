import numpy as np
import pickle
from scipy import io
import torch
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import TensorDataset, DataLoader
from scipy import signal
devices = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def DBnorm(X):
    for ii in range(0, np.size(X,0)):
        temp_data = torch.Tensor(X[ii, :, :])
        norm_data = torch.nn.functional.normalize(temp_data, p=2.0, dim=1, eps=1e-12, out=None)
        X[ii, :, :] = norm_data
        del norm_data
    return X

def DBnorm_2D(X):
    for ii in range(0, np.size(X,0)):
        temp_data = torch.Tensor(X[ii, :])
        norm_data = torch.nn.functional.normalize(temp_data, p=2.0, dim=0, eps=1e-12, out=None)
        X[ii, :] = np.array(norm_data)
        del norm_data
    return X

def TacDataset(Dir, subidx, random_seed):
    default_dir = Dir
    file_x_path = default_dir / 'X.mat'
    file_y_path = default_dir / 'Y_{}.mat'.format(subidx)
    Dat_X = io.loadmat(file_x_path)
    Dat = io.loadmat(file_y_path)
    X = Dat_X['X']
    Y_obj = Dat['Y'][:,0]
    Y_percep = Dat['Y'][:,1]
    
    obj_list = [X[Y_obj == obj_idx] for obj_idx in range(np.max(Y_obj) + 1)]
    label_list = [Y_percep[Y_obj == obj_idx] for obj_idx in range(np.max(Y_obj) + 1)]
    Y_label = np.array([label_list[i][0] for i in range(np.max(Y_obj) + 1)])
    
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_seed)
    Dataset = [None] * 5
    n = 0
    for temp_index, test_index in kf.split(obj_list, Y_label):
        X_temp = [obj_list[i] for i in temp_index]
        Y_temp = [label_list[i] for i in temp_index]
        Y_tlabel = np.array([Y_temp[i][0] for i in range(0,len(Y_temp))])
        
        X_test = np.vstack([obj_list[i] for i in test_index])
        Y_test = np.hstack([label_list[i] for i in test_index])
        
        kf2 = StratifiedKFold(n_splits=4, shuffle=True, random_state=random_seed)
        for train_index, val_index in kf2.split(X_temp, Y_tlabel):
            X_train = np.vstack([X_temp[i] for i in train_index])
            Y_train = np.hstack([Y_temp[i] for i in train_index])
            shuffle_array = np.arange(Y_train.size)
            np.random.shuffle(shuffle_array)
            X_train = X_train[shuffle_array,:]
            Y_train = Y_train[shuffle_array]
              
            X_val = np.vstack([X_temp[i] for i in val_index])
            Y_val = np.hstack([Y_temp[i] for i in val_index])
            break
        
        Dataset[n] = torch.tensor(DBnorm_2D(X_train), dtype=torch.float32), torch.tensor(Y_train, dtype=torch.int64), torch.tensor(DBnorm_2D(X_val), dtype=torch.float32), torch.tensor(Y_val, dtype=torch.int64), torch.tensor(DBnorm_2D(X_test), dtype=torch.float32), torch.tensor(Y_test, dtype=torch.int64)
        n = n+1
    
    return Dataset