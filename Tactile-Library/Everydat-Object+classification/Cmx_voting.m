clear all
close all
clc
Working_dir = 'E:\Project\NRF_Tactile-Fabric\2025\Fabric_time\Codes\Regression+time+TactNet_alldatatype_FBbranch\log';
time_list = {'manual','manual_L','manual_R', 'multi', 'multi_L','multi_R', 'crop_2s','crop_4s', 'crop_8s'};
mode_list = {'time','branch','hybrid'};
act_list = {'identity', 'scaled_sigmoid', 'scaled_tanh'};
model_list = {'TacNet'};
fb_list = [3, 5, 7];

for mdl_idx = 1
    for ty_idx = 2:2
        for mode_idx = 0:2
            for act_idx = 1:1
                for fb_idx = 1:3
                    for iter_idx = 1:5
                        tot_te_gts = [];
                        tot_te_pred1 = [];
                        new_tot_te_gts = [];
                        new_tot_te_pred1 = [];
                        tot_te_acc1 = [];
                        tot_tr_gts = [];
                        tot_tr_pred1 = [];
                        new_tot_tr_gts = [];
                        new_tot_tr_pred1 = [];
                        tot_tr_acc1 = [];
                        k = 5*(iter_idx-1);
                        for fold_idx = 0:4
                            % load([Working_dir, '\Reg_' model_list{mdl_idx} '_' num2str(fold_idx) '_revise_label.mat']
                            load([Working_dir, '\Reg_' model_list{mdl_idx}...
                                '_' num2str(k+fold_idx) '_' num2str(ty_idx)...
                                '_' mode_list{mode_idx+1} '_' act_list{act_idx+1}...
                                '_' num2str(fb_list(fb_idx)) '.mat'])
                            tot_tr_gts = [tot_tr_gts, train_true];
                            tot_tr_pred1 = [tot_tr_pred1, train_pred];
                            tot_te_gts = [tot_te_gts, test_true];
                            tot_te_pred1 = [tot_te_pred1, test_pred];
                        end
                        k1 = unique(tot_tr_gts);
                        disp(length(k1))
                        for i = 1:length(k1)
                            temp_idx = (tot_tr_gts == k1(i));
                            temp_sample1 = tot_tr_gts(temp_idx);
                            temp_sample2 = tot_tr_pred1(temp_idx);
                            new_tot_tr_gts_temp = mean(temp_sample1);
                            new_tot_tr_pred_temp = mean(temp_sample2);
                            clear temp_sample1 temp_sample2 temp_idx
                            new_tot_tr_gts = [new_tot_tr_gts, new_tot_tr_gts_temp];
                            new_tot_tr_pred1 = [new_tot_tr_pred1, new_tot_tr_pred_temp];
                        end
                        k2 = unique(tot_te_gts);
                        disp(length(k2))
                        for i = 1:length(k2)
                            temp_idx = (tot_te_gts == k2(i));
                            temp_sample1 = tot_te_gts(temp_idx);
                            temp_sample2 = tot_te_pred1(temp_idx);
                            new_tot_te_gts_temp = mean(temp_sample1);
                            new_tot_te_pred_temp = mean(temp_sample2);
                            clear temp_sample1 temp_sample2 temp_idx
                            new_tot_te_gts = [new_tot_te_gts, new_tot_te_gts_temp];
                            new_tot_te_pred1 = [new_tot_te_pred1, new_tot_te_pred_temp];
                        end
                        corr_tr(fb_idx, mode_idx+1, iter_idx) = corr2(tot_tr_gts,tot_tr_pred1);
                        corr_te(fb_idx, mode_idx+1, iter_idx) = corr2(tot_te_gts,tot_te_pred1);
                        rmse_tr(fb_idx, mode_idx+1, iter_idx) = calc_rmse(tot_tr_gts,tot_tr_pred1);
                        rmse_te(fb_idx, mode_idx+1, iter_idx) = calc_rmse(tot_te_gts,tot_te_pred1);
                        R2_tr(fb_idx, mode_idx+1, iter_idx) = calc_r2_score(tot_tr_gts,tot_tr_pred1);
                        R2_te(fb_idx, mode_idx+1, iter_idx) = calc_r2_score(tot_te_gts,tot_te_pred1);
                        mae_tr(fb_idx, mode_idx+1, iter_idx) = calc_mae(tot_tr_gts,tot_tr_pred1);
                        mae_te(fb_idx, mode_idx+1, iter_idx) = calc_mae(tot_te_gts,tot_te_pred1);

                        corr_tr2(fb_idx, mode_idx+1, iter_idx) = corr2(new_tot_tr_gts,new_tot_tr_pred1);
                        corr_te2(fb_idx, mode_idx+1, iter_idx) = corr2(new_tot_te_gts,new_tot_te_pred1);
                        rmse_tr2(fb_idx, mode_idx+1, iter_idx) = calc_rmse(new_tot_tr_gts,new_tot_tr_pred1);
                        rmse_te2(fb_idx, mode_idx+1, iter_idx) = calc_rmse(new_tot_te_gts,new_tot_te_pred1);
                        R2_tr2(fb_idx, mode_idx+1, iter_idx) = calc_r2_score(new_tot_tr_gts,new_tot_tr_pred1);
                        R2_te2(fb_idx, mode_idx+1, iter_idx) = calc_r2_score(new_tot_te_gts,new_tot_te_pred1);
                        mae_tr2(fb_idx, mode_idx+1, iter_idx) = calc_mae(new_tot_tr_gts,new_tot_tr_pred1);
                        mae_te2(fb_idx, mode_idx+1, iter_idx) = calc_mae(new_tot_te_gts,new_tot_te_pred1);

                        % figure(1)
                        % subplot(1,2,1)
                        % scatter(tot_tr_gts,tot_tr_pred1)
                        % subplot(1,2,2)
                        % scatter(tot_te_gts,tot_te_pred1)
                        % disp([model_list{mod_idx} '_' time_list{ty_idx} '_' mode_list{mode_idx+1} '_' act_list{act_idx+1}])

                        % t=6, b=4, h=6
                        % figure(2)
                        % subplot(1,2,1)
                        % scatter(new_tot_tr_gts,new_tot_tr_pred1,'filled','MarkerFaceColor',[0 0 0])
                        % subplot(1,2,2)
                        % scatter(new_tot_te_gts,new_tot_te_pred1,'filled','MarkerFaceColor',[0 0 0])
                        % disp([model_list{mdl_idx} '_' time_list{ty_idx} '_' mode_list{mode_idx+1} '_' act_list{act_idx+1}])
                    end
                end
            end
        end
    end
end

% [p(1),h(1)] = signrank(corr_te(1,:),corr_te(2,:));
% [p(2),h(2)] = signrank(corr_te(1,:),corr_te(3,:));
% [p(3),h(3)] = signrank(corr_te(2,:),corr_te(3,:));
% [~, ~, ~, adj_p] = fdr_bh(p);

mcorr_tr = mean(corr_tr,3);
mcorr_te = mean(corr_te,3);
mrmse_tr = mean(rmse_tr,3);
mrmse_te = mean(rmse_te,3);
mR2_tr = mean(R2_tr,3);
mR2_te = mean(R2_te,3);
mmae_tr = mean(mae_tr,3);
mmae_te = mean(mae_te,3);

mcorr_tr2 = mean(corr_tr2,3);
mcorr_te2 = mean(corr_te2,3);
mrmse_tr2 = mean(rmse_tr2,3);
mrmse_te2 = mean(rmse_te2,3);
mR2_tr2 = mean(R2_tr2,3);
mR2_te2 = mean(R2_te2,3);
mmae_tr2 = mean(mae_tr2,3);
mmae_te2 = mean(mae_te2,3);
clearvars -except mmae_te mmae_tr mcorr_tr mcorr_te mrmse_tr mrmse_te mR2_tr mR2_te...
    mmae_te2 mmae_tr2 mcorr_tr2 mcorr_te2 mrmse_tr2 mrmse_te2 mR2_tr2 mR2_te2
