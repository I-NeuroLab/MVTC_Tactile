import numpy as np
import pickle
from scipy import io
import torch
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import TensorDataset, DataLoader
from scipy import signal
devices = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from sklearn.preprocessing import StandardScaler

def TacDataset(Dir, random_seed):
    default_dir = Dir
    file_x_path = default_dir / 'X_63.mat'
    file_y_path = default_dir / 'Y_avg_63.mat'
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
        
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_val   = scaler.transform(X_val)
        X_test  = scaler.transform(X_test)
        
        Dataset[n] = torch.tensor(X_train, dtype=torch.float32), torch.tensor(Y_train, dtype=torch.float32), torch.tensor(X_val, dtype=torch.float32), torch.tensor(Y_val, dtype=torch.float32), torch.tensor(X_test, dtype=torch.float32), torch.tensor(Y_test, dtype=torch.float32)
        n = n+1
    
    return Dataset