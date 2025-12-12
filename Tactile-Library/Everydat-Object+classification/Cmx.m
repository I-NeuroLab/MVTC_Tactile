clear all
close all
clc
Working_dir = 'E:\Project\IITP_AGI_MVTC\Codes\classification+time+TactNet+FBbranch+50class\log';
mode_list = {'time','branch','hybrid'};
model_list = {'TacNet'};
feat_num = 10;

for mode_idx = 2
    for feat_idx = 1:length(feat_num)
        for iter_idx = 1:1
            tot_te_gts = [];
            tot_te_pred1 = [];
            tot_te_acc1 = [];
            tot_tr_gts = [];
            tot_tr_pred1 = [];
            tot_tr_acc1 = [];
            k = 5*(iter_idx-1);
            for fold_idx = 0:4
                load([Working_dir, '\Reg_' num2str(k+fold_idx) ...
                    '_' mode_list{mode_idx+1} '_identity'...
                    '_fb3_y' num2str(feat_num(feat_idx)) '_fv.mat'])
                [~, trpred_idx] = max(train_pred, [], 2);
                [~, tepred_idx] = max(test_pred, [], 2);

                tot_tr_gts = [tot_tr_gts, train_true];
                tot_tr_pred1 = [tot_tr_pred1; trpred_idx-1];
                tot_te_gts = [tot_te_gts, test_true];
                tot_te_pred1 = [tot_te_pred1; tepred_idx-1];
            end
            cfm = confusionmat(tot_te_gts,int64(tot_te_pred1)');
            acc_te(feat_idx, iter_idx) = sum(diag(cfm))/length(tot_te_gts);

        end
    end
end

macc_te = mean(acc_te,2);