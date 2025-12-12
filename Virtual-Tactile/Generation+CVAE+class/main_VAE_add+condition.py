
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
    
    devices = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #for rep in range(0,5):
    for rep in range(5,10):
        label_idx = rep+1
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
        del Temp_data[1:]
        for temp in zip(Temp_data):
            
            X_train, Y_train, M_train, X_val, Y_val, M_val, X_test, Y_test, M_test = temp[0]
            X_train = X_train.unsqueeze(1)
            X_val = X_val.unsqueeze(1)
            X_test = X_test.unsqueeze(1)
            Y_train = torch.round(Y_train, decimals=0)
            Y_val = torch.round(Y_val, decimals=0)
            Y_test = torch.round(Y_test, decimals=0)
    
            train_dataset = TensorDataset(X_train, Y_train, M_train)
            val_dataset = TensorDataset(X_val, Y_val, M_val)
            test_dataset = TensorDataset(X_test, Y_test, M_test)
    
            train_dataloader = DataLoader(
                train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
            val_dataloader = DataLoader(
                val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
            test_dataloader = DataLoader(
                test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
            
            model = VAE(input_dim=5000, latent_dim=latent, device=devices, num_classes = 6, num_material_classes=10).to(devices)
            
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=args.lr, weight_decay=5e-5)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.epochs)
    
            # Track best results for this fold
            best_loss = float('inf')
            best_model_state = None
    
            # Training loop for this fold
            early_stopper = EarlyStopping(patience=10, delta=0.001)
            
            for epoch in range(args.epochs):
                tr_loss, tr_acc1, tr_acc2 = train_one_epoch_VAE(model, train_dataloader, optimizer)
                val_loss, val_acc1, val_acc2 = validation_VAE(model, val_dataloader)
                torch.cuda.empty_cache()
    
                print("\tEpoch_tr", epoch + 1, f"\tAverage Loss: {tr_loss:.4f}", f"\tAverage Acc1: {tr_acc1:.4f}", f"\tAverage Acc2: {tr_acc2:.4f}")
                
                print("\tEpoch_val", epoch + 1, f"\tAverage Loss: {val_loss:.4f}", f"\tAverage Acc1: {val_acc1:.4f}", f"\tAverage Acc2: {val_acc2:.4f}")
                
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
    
            ckpt_path = save_dir / f"Best_set{rep}_fold{CV_cnt}_ver2.pt"
            torch.save({"state_dict": best_model_state}, ckpt_path)
            del model
            
            ckpt_path = save_dir / f"Best_set{rep}_fold{CV_cnt}_ver2.pt"
            checkpoint = torch.load(ckpt_path, map_location=devices)
            best_model_state = checkpoint["state_dict"]
    
            model = VAE(input_dim=5000, latent_dim=latent, device=devices, num_classes = 6, num_material_classes=10).to(devices)
            model.load_state_dict(best_model_state)
                            
            tr_loss, tr_acc1, tr_acc2 = validation_VAE(model,test_dataloader)
            val_loss, val_acc1, val_acc2 = validation_VAE(model,test_dataloader)
            te_loss, te_acc1, te_acc2 = validation_VAE(model,test_dataloader)
            tr_latent_vectors, tr_mat, tr_label, tr_dat, tr_recon = extract_latent(model, train_dataloader)
            val_latent_vectors, val_mat, val_label, val_dat, val_recon = extract_latent(model, val_dataloader)
            te_latent_vectors, test_mat, test_label, te_dat, te_recon = extract_latent(model, test_dataloader)
            
            pseuro_te_latent_vectors, pseuro_test_mat, pseuro_test_label, pseuro_te_recon = extract_latent_valid(model, num_samples_per_class=1)
            print( f"\ttr_loss: {tr_loss:.4f}", f"\ttr_acc1: {te_acc1:.4f}", f"\ttr_acc2: {te_acc2:.4f}")
            print( f"\tval_loss: {val_loss:.4f}", f"\tval Acc1: {val_acc2:.4f}", f"\tval Acc2: {val_acc2:.4f}")
            print( f"\tte_loss: {te_loss:.4f}", f"\tte Acc1: {te_acc1:.4f}", f"\tte Acc2: {te_acc2:.4f}")
            
            savedict = {
                'tr_loss': tr_loss,
                'tr_acc1': tr_acc1,
                'tr_acc2': tr_acc2,
                'tr_latent_vectors': tr_latent_vectors,
                'tr_label': tr_label,
                'tr_mat': tr_mat,
                'tr_dat': tr_dat,
                'tr_recon': tr_recon,
                
                'val_loss': val_loss,
                'val_acc1': val_acc1,
                'val_acc2': val_acc2,
                'val_latent_vectors': val_latent_vectors,
                'val_label': val_label,
                'val_mat': val_mat,
                'val_dat': val_dat,
                'val_recon': val_recon,
                
                'te_loss': te_loss,
                'te_acc1': te_acc1,
                'te_acc2': te_acc2,
                'te_latent_vectors': te_latent_vectors,
                'te_label': test_label,
                'test_mat': test_mat,
                'te_dat': te_dat,
                'te_recon': te_recon,
                
                'pseuro_te_latent_vectors': pseuro_te_latent_vectors,
                'pseuro_te_label': pseuro_test_label,
                'pseuro_test_mat': pseuro_test_mat,
                'pseuro_te_recon': pseuro_te_recon
            }
    
            save_str = save_dir / f"VAE_set{rep}_fold{CV_cnt}_ver2.mat"
            io.savemat(save_str, savedict)
            CV_cnt = CV_cnt+1
        
if __name__ == '__main__':

    args = parse_argument()
    subject_results = main(args)
