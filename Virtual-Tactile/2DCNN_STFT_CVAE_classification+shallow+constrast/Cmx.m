clear all
close all
clc
Working_dir = 'D:\LHT\Code\2DCNN_STFT_CVAE_regression+shallow\log';

for sub_idx = 1:9
    tot_te_gts = [];
    tot_te_pred1 = [];
    tot_te_pred2 = [];

    for fold_idx = 0:4
        load([Working_dir, '\Clf' num2str(sub_idx) '_' num2str(fold_idx) '_al(0.5)_be(2.0)_gm(2.0).mat'])
        tot_te_gts = [tot_te_gts, test_gts];
        tot_te_pred1 = [tot_te_pred1, test_preds_latent];
        tot_te_pred2 = [tot_te_pred2, test_preds_time];
        tot_te_mse(fold_idx+1) = test_mse;
    end
    figure(1)
    subplot(3,3,sub_idx)
    scatter(tot_te_gts,tot_te_pred1);
    figure(2)
    subplot(3,3,sub_idx)
    scatter(tot_te_gts,tot_te_pred2);
    sub_corr1(sub_idx) = corr2(tot_te_gts,tot_te_pred1);
    sub_corr2(sub_idx) = corr2(tot_te_gts,tot_te_pred2);
    sub_te_mse1(sub_idx) = mse(tot_te_gts, tot_te_pred1);
    sub_te_mse2(sub_idx) = mse(tot_te_gts, tot_te_pred2);
end