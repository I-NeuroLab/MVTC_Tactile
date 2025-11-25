
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

def main(args):
    
    save_dir = (Path(__file__).resolve().parent / "log")
    save_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    args.epochs = 200
    args.batch_size = 32
    latent = 1000
    label_idx = 10
    
    devices = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CV_cnt = 0
    for rep in range(0,2):
        Temp_data = TacDataset_save(args.data_dir,
                                    save_dir,
                                    label_idx,
                                    args.seed,
                                    zscore_mode="per_channel_global",
                                    shuffle_arrays=False,
                                    save_stats=True,
                                    n_repeats=1,
                                    repeat_idx=rep,
                                    splits_root=save_dir)

        # Perform 5-fold cross validation
        CV_cnt = 0
        for temp in zip(Temp_data):
            
            X_train, Y_train, X_val, Y_val, X_test, Y_test = temp[0]
            X_train = X_train.unsqueeze(1)
            X_val = X_val.unsqueeze(1)
            X_test = X_test.unsqueeze(1)
            Y_train = torch.round(Y_train, decimals=0)
            Y_val = torch.round(Y_val, decimals=0)
            Y_test = torch.round(Y_test, decimals=0)
    
            train_dataset = TensorDataset(X_train, Y_train)
            val_dataset = TensorDataset(X_val, Y_val)
            test_dataset = TensorDataset(X_test, Y_test)
    
            train_dataloader = DataLoader(
                train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
            val_dataloader = DataLoader(
                val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
            test_dataloader = DataLoader(
                test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
            
            model = VAE(input_dim=5000, latent_dim=1000, device=devices, num_classes = 6).to(devices)
            
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=args.lr, weight_decay=5e-5)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.epochs)
    
            # Track best results for this fold
            best_loss = float('inf')
            best_model_state = None
    
            # Training loop for this fold
            early_stopper = EarlyStopping(patience=20, delta=0.001)
            for epoch in range(args.epochs):
                tr_loss, tr_acc = train_one_epoch_VAE(model, train_dataloader, optimizer)
                val_loss, val_acc = validation_VAE(model, val_dataloader)
                torch.cuda.empty_cache()
    
                print("\tEpoch_tr", epoch + 1, f"\tAverage Loss: {tr_loss:.4f}", f"\tAverage Acc: {tr_acc:.4f}")
                
                print("\tEpoch_val", epoch + 1, f"\tAverage Loss: {val_loss:.4f}", f"\tAverage Acc: {val_acc:.4f}")
                
                scheduler.step()
    
                model_updated = False
                if best_loss > val_loss:
                    best_loss = val_loss
                    best_model_state = copy.deepcopy(model.state_dict())
                    model_updated = True
                    print("Model updated")
    
                early_stopper(val_loss)
                if early_stopper.early_stop:
                    print("Early stopping triggered at epoch", epoch + 1)
                    break
    
            torch.save(model.state_dict(), save_dir / f'VAE_weight_{CV_cnt}.pth')
            
            te_loss, te_acc = validation_VAE(model,test_dataloader)
            tr_latent_vectors, tr_label, tr_dat, tr_recon = extract_latent(model, train_dataloader)
            val_latent_vectors, val_label, val_dat, val_recon = extract_latent(model, val_dataloader)
            te_latent_vectors, test_label, te_dat, te_recon = extract_latent(model, test_dataloader)
            
            pseuro_val_latent_vectors, pseuro_val_label = extract_latent_valid(model, val_dataloader)
            pseuro_te_latent_vectors, pseuro_test_label = extract_latent_valid(model, test_dataloader)
            print( f"\ttr_loss: {tr_loss:.4f}", f"\ttr Acc: {tr_acc:.4f}")
            print( f"\tval_loss: {val_loss:.4f}", f"\tval Acc: {val_acc:.4f}")
            print( f"\tte_loss: {te_loss:.4f}", f"\tte Acc: {te_acc:.4f}")
            
            savedict = {
                'tr_loss': tr_loss,
                'tr_acc': tr_acc,
                'tr_latent_vectors': tr_latent_vectors,
                'tr_label': tr_label,
                'tr_dat': tr_dat,
                'tr_recon': tr_recon,
                
                'val_loss': val_loss,
                'val_acc': val_acc,
                'val_latent_vectors': val_latent_vectors,
                'val_label': val_label,
                'val_dat': val_dat,
                'val_recon': val_recon,
                'pseuro_val_latent_vectors': pseuro_val_latent_vectors,
                'pseuro_val_label': pseuro_val_label,
                
                'te_loss': te_loss,
                'te_acc': te_acc,
                'te_latent_vectors': te_latent_vectors,
                'test_label': test_label,
                'te_dat': te_dat,
                'te_recon': te_recon,
                'pseuro_te_latent_vectors': pseuro_te_latent_vectors,
                'pseuro_test_label': pseuro_test_label
            }
    
            save_str = save_dir + \
                'VAE_{}_noshuffle_latent.mat'.format(CV_cnt)
            io.savemat(save_str, savedict)
        
if __name__ == '__main__':

    args = parse_argument()
    subject_results = main(args)
