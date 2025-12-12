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

def normalize_target_shape(target: torch.Tensor) -> torch.Tensor:
    if target.dim() == 1:
        return target.view(-1)
    elif target.dim() == 2 and target.size(1) == 1:
        return target.view(-1)
    else:
        return target
    
def train_one_epoch(model, dataloader, optimizer):
    model.train()
    overall_loss = 0.0
    all_preds, all_true = [], []
    first_batch_print = True

    for inputs, targets in dataloader:
        inputs = inputs.to(devices)
        targets = normalize_target_shape(targets)
        targets = targets.long().to(devices)
        
        optimizer.zero_grad()

        preds = model(inputs)
        preds = normalize_target_shape(preds)
        targets = targets.to(preds.device, non_blocking=True)
        
        preds = F.log_softmax(preds, dim=1)
        loss = F.cross_entropy(preds, targets)
        loss.backward()
        optimizer.step()

        overall_loss += loss.item()
        all_preds.append(preds.detach().cpu())
        all_true.append(targets.detach().cpu())

    avg_loss = overall_loss / max(1, len(dataloader))
    preds_cat = torch.cat(all_preds).squeeze().cpu().detach().numpy()
    true_cat  = torch.cat(all_true).squeeze().cpu().detach().numpy()

    acc = (np.argmax(preds_cat,1) == true_cat).sum()/len(true_cat)

    return avg_loss, preds_cat, true_cat, acc

@torch.no_grad()
def validation(model, dataloader):
    model.eval()
    overall_loss = 0.0
    all_preds, all_true = [], []

    for inputs, targets in dataloader:
        inputs = inputs.to(devices)
        targets = normalize_target_shape(targets)
        targets = targets.long().to(devices)
        
        preds = model(inputs)
        preds = normalize_target_shape(preds)
        targets = targets.to(preds.device, non_blocking=True)
        
        preds = F.log_softmax(preds, dim=1)
        loss = F.cross_entropy(preds, targets)
        overall_loss += loss.item()

        all_preds.append(preds.detach().cpu())
        all_true.append(targets.detach().cpu())

    avg_loss = overall_loss / max(1, len(dataloader))
    preds_cat = torch.cat(all_preds).squeeze().cpu().detach().numpy()
    true_cat  = torch.cat(all_true).squeeze().cpu().detach().numpy()

    acc = (np.argmax(preds_cat,1) == true_cat).sum()/len(true_cat)

    return avg_loss, preds_cat, true_cat, acc
