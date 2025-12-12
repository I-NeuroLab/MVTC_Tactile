
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
label_list = [10]

def main(args):
    model = 'TacNet'
    for label_idx in label_list:
        for mode_idx in range(2,3):
            print("label_idx = {}".format(label_idx))
            save_dir = (Path(__file__).resolve().parent / "log")
            save_dir.mkdir(parents=True, exist_ok=True)
            modelname = model
            random.seed(args.seed)
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)
            torch.cuda.manual_seed(args.seed)
            args.epochs = 1000
            args.batch_size = 64 #64
            args.lr = 5e-4
            args.pool_stride = 32 #32
            args.K = 1 #1
            args.use_aug = False
            args.share_subband_conv = False
            args.fbank_n = 3
            args.output_activation = "identity"
            if label_idx == 15:
                args.n_class = 14
            else:
                args.n_class = 50
            fbank_bounds = {
                "SA1": (2.0, 32.0),
                "SA2": (0.0, 8.0),
                "RA1": (5.0, 64.0),
                "RA2": (64.0, 400.0),
            }
            args.fbank_custom = make_fbank_index(fbank_n=args.fbank_n,
                                                bounds_per_receptor=fbank_bounds)
            
            if mode_idx == 0:
                args.mode = "time"
            elif mode_idx == 1:
                args.mode = "branch"
            else:
                args.mode = "hybrid"
                
            devices = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            CV_cnt = 0
            for rep in range(0,1):
                Temp_data = TacDataset_save(args.data_dir,
                                            save_dir,
                                            label_idx,
                                            args.seed,
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
                        d_model=32, nhead=4, num_layers=2, dim_feedforward=256,
                        dropout=0.3, mode=args.mode,
                        output_activation=args.output_activation, output_minmax=(-10.0, 10.0), n_class=args.n_class,
                        fbank_n=getattr(args, "fbank_n", 3),
                        fbank_custom=getattr(args, "fbank_custom", None),
                        share_subband_conv=getattr(args, "share_subband_conv", False),
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
                        tr_loss, _, _, tr_acc = train_one_epoch(
                            model, train_dataloader, optimizer)
                        val_loss, _, _, val_acc = validation(
                            model, val_dataloader)
                        torch.cuda.empty_cache()
            
                        print("\tEpoch_tr", epoch + 1, f"\tAverage Loss: {tr_loss:.4f}",
                              f"\ttr_acc: {tr_acc:.4f}")
                        
                        print("\tEpoch_val", epoch + 1, f"\tAverage Loss: {val_loss:.4f}",
                              f"\tval_acc: {val_acc:.4f}")
                        
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
            
                    ckpt_path = save_dir / f"Best_{CV_cnt}_{args.mode}_{args.output_activation}_fb{args.fbank_n}_y{label_idx}_fv.pt"
                    torch.save({"state_dict": best_model_state, "model_kwargs": model_kwargs}, ckpt_path)
    
                    del model
                    model = TactileRoughnessHybridNet(**model_kwargs).to(devices)
                    model.load_state_dict({k: v.to(devices) for k, v in best_model_state.items()})
                                
                    tr_loss, tr_pred, tr_true, tr_acc = validation(
                        model, train_dataloader)
                    val_loss, val_pred, val_true, val_acc = validation(
                        model, val_dataloader)
                    te_loss, te_pred, te_true, te_acc = validation(
                        model, test_dataloader)
                    torch.cuda.empty_cache()
                    
                    print( f"\ttr_loss: {tr_loss:.4f}",
                          f"\ttr_acc: {tr_acc:.4f}")
                    
                    print( f"\tval_loss: {val_loss:.4f}",
                          f"\tval_acc: {val_acc:.4f}")
                    
                    print( f"\tte_loss: {te_loss:.4f}",
                          f"\tte_acc: {te_acc:.4f}")
                    """
                    # Save attention (mean over heads & batches)
                    attn_mean_layers, attn_meta = compute_mean_attention_over_loader(model, test_dataloader, devices, mode=args.mode)
                    attn_path = save_dir / f"Attn_{CV_cnt}_{args.mode}_{args.output_activation}_fb{args.fbank_n}_y{label_idx}.mat"
                    io.savemat(str(attn_path), {
                        "attn_mean_layers": attn_mean_layers,  # (L,S,S)
                        "attn_L": attn_mean_layers.shape[0],
                        "attn_S": attn_mean_layers.shape[1],
                        "mode": args.mode
                    })
                    
                    baseprefix = f"Attn_{CV_cnt}_{args.mode}_{args.output_activation}_fb{args.fbank_n}_y{label_idx}"
                    save_per_token_importance(save_dir, baseprefix, attn_mean_layers, attn_meta, model_kwargs, T=int(X_train.shape[2]))
                    """
                    savedict = {
                        'train_loss': tr_loss,
                        'train_pred': tr_pred,
                        'train_true': tr_true,
                        'train_acc': tr_acc,
                        
                        'val_loss': val_loss,
                        'val_pred': val_pred,
                        'val_true': val_true,
                        'val_acc': val_acc,
                        
                        'test_loss': te_loss,
                        'test_pred': te_pred,
                        'test_true': te_true,
                        'test_acc': te_acc
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
                    
                    hist_path = save_dir / f'History_{CV_cnt}_{args.mode}_{args.output_activation}_fb{args.fbank_n}_y{label_idx}_fv.csv'
                    reg_path  = save_dir / f'Reg_{CV_cnt}_{args.mode}_{args.output_activation}_fb{args.fbank_n}_y{label_idx}_fv.mat'
                    history.to_csv(hist_path, index=False)
                    io.savemat(str(reg_path), savedict)
                    del history, savedict
                    print("\tSavedir", save_dir)
                    CV_cnt = CV_cnt+1
    
if __name__ == '__main__':

    args = parse_argument()
    subject_results = main(args)
