
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

def main(args):

    save_dir = 'E:\\Project\\NRF_Tactile-standardization\\2025\\New_segment\\Codes\\2DCNN_STFT_CVAE_classification+shallow+constrast\\log\\'
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    args.epochs = 150
    args.batch_size = 64
    latent = 1000
    alpha = 1
    beta = 1
    theta = 1
    
    devices = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for subidx in range(1, 10):
        Temp_data = TacDataset(args.data_dir, subidx, args.seed)

        # Perform 5-fold cross validation
        CV_cnt = 0
        for temp in zip(Temp_data):
            
            history = pd.DataFrame(columns=[
                'epoch', 'training_loss', 'validation_loss',
                'tr_latent_mse', 'val_latent_mse',
                'tr_time_mse', 'val_time_mse',
                'model_updated',
                'gts_mes_tr', 'gts_mes_val', 'gts_mes_test'
            ])

            X_train, Y_train, X_val, Y_val, X_test, Y_test = temp[0]

            spectrogram = torchaudio.transforms.Spectrogram(
                n_fft=5000,
                win_length=5000,
                hop_length=30,
                power=2.0
            ).to(devices)

            spec = spectrogram(X_test.to(devices))
            spec = spec[:, 0:300, :].unsqueeze(1)

            train_dataset = TensorDataset(X_train, Y_train)
            val_dataset = TensorDataset(X_val, Y_val)
            test_dataset = TensorDataset(X_test, Y_test)

            train_dataloader = DataLoader(
                train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
            val_dataloader = DataLoader(
                val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
            test_dataloader = DataLoader(
                test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)

            model = VAE(
                input_shape=spec.shape[1:], latent_dim=latent, device=devices, y_dim=1).to(devices)

            optimizer = torch.optim.AdamW(
                model.parameters(), lr=args.lr, weight_decay=5e-5)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.epochs)

            # Track best results for this fold
            best_loss = float('inf')
            best_accs = 0.
            best_model_state = None

            # Training loop for this fold
            early_stopper = EarlyStopping(patience=20, delta=0.001)
            for epoch in range(args.epochs):
                tr_loss, tr_acc1, tr_acc2, ttr_acc1, ttr_acc2, ttr_acc3 = train_one_epoch(
                    model, train_dataloader, optimizer, alpha, beta, theta)
                val_loss, val_acc1, val_acc2, tval_acc1, tval_acc2, tval_acc3 = validation(
                    model, val_dataloader, alpha, beta, theta)

                print("\tEpoch_tr", epoch + 1, f"\tAverage Loss: {tr_loss:.4f}",
                      f"\tLatent Acc: {tr_acc1:.4f}", f"\tTime Acc: {tr_acc2:.4f}")
                print("\tEpoch_tr", epoch + 1, f"\tLatent Acc: {ttr_acc1:.4f}",
                      f"\tTime Acc: {ttr_acc2:.4f}", f"\tHybrid Acc: {ttr_acc2:.4f}")
                
                print("\tEpoch_val", epoch + 1, f"\tAverage Loss: {val_loss:.4f}",
                      f"\tLatent Acc: {val_acc1:.4f}", f"\tTime Acc: {val_acc2:.4f}")
                print("\tEpoch_val", epoch + 1, f"\tLatent Acc: {tval_acc1:.4f}",
                      f"\tTime Acc: {tval_acc2:.4f}", f"\tHybrid Acc: {tval_acc3:.4f}")
                

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
                    tr_acc1,
                    val_acc1,
                    tr_acc2,
                    val_acc2,
                    model_updated,
                    0,
                    0,
                    0
                ]

                early_stopper(val_loss)
                if early_stopper.early_stop:
                    print("Early stopping triggered at epoch", epoch + 1)
                    break

            del model
            model = VAE(
                input_shape=spec.shape[1:], latent_dim=latent, device=devices, y_dim=1).to(devices)
            model.load_state_dict(best_model_state)

            preds1_train, preds2_train, gts_train, plabel1_train, plabel2_train, plabel3_train= predict_conditional_vae(
                model, train_dataloader)
            preds1_val, preds2_val, gts_val, plabel1_val, plabel2_val, plabel3_val = predict_conditional_vae(
                model, val_dataloader)
            preds1_test, preds2_test, gts_test, plabel1_test, plabel2_test, plabel3_test= predict_conditional_vae(
                model, test_dataloader)

            print("\tEpoch_te", epoch + 1, f"\tacc1_train: {plabel1_train:.4f}", f"\tacc2_train: {plabel2_train:.4f}",
                  f"\tacc3_train: {plabel3_train:.4f}",
                  f"\tacc1_val: {plabel1_val:.4f}", f"\tacc2_val: {plabel2_val:.4f}", f"\tacc3_val: {plabel3_val:.4f}",
                  f"\tacc1_test: {plabel1_test:.4f}", f"\tacc2_test: {plabel2_test:.4f}", f"\tacc3_test: {plabel3_test:.4f}")
            
            savedict = {
                'train_preds_latent': preds1_train.cpu().detach().numpy(),
                'train_preds_time': preds2_train.cpu().detach().numpy(),
                'train_acc_latent': plabel1_train.cpu().detach().numpy(),
                'train_acc_time': plabel2_train.cpu().detach().numpy(),
                'train_gts': gts_train.cpu().detach().numpy(),
                
                'val_preds_latent': preds1_val.cpu().detach().numpy(),
                'val_preds_time': preds2_val.cpu().detach().numpy(),
                'val_acc_latent': plabel1_val.cpu().detach().numpy(),
                'val_acc_time': plabel2_val.cpu().detach().numpy(),
                'val_gts': gts_val.cpu().detach().numpy(),
                
                'test_preds_latent': preds1_test.cpu().detach().numpy(),
                'test_preds_time': preds2_test.cpu().detach().numpy(),
                'test_acc_latent': plabel1_test.cpu().detach().numpy(),
                'test_acc_time': plabel2_test.cpu().detach().numpy(),
                'test_gts': gts_test.cpu().detach().numpy(),
            }

            history.loc[len(history)] = [
                epoch + 2,
                tr_loss,
                val_loss,
                tr_acc1,
                val_acc1,
                tr_acc2,
                val_acc2,
                False,
                plabel2_train.cpu().detach().numpy(),
                plabel2_val.cpu().detach().numpy(),
                plabel2_test.cpu().detach().numpy()
            ]
            history.to_csv(
                save_dir + 'training_history_{}_{}_al({})_be({})_gm({}).csv'.format(subidx, CV_cnt, alpha, beta, theta), index=False)
            
            save_str = save_dir + \
                'Clf_{}_{}_al({})_be({})_gm({}).mat'.format(subidx, CV_cnt, alpha, beta, theta)
            io.savemat(save_str, savedict)
            CV_cnt = CV_cnt+1
            del history, savedict
            print("\tSavedir", save_dir)

if __name__ == '__main__':

    args = parse_argument()
    subject_results = main(args)
