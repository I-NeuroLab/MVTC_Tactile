clear all
close all
clc
Working_dir = 'E:\Project\NRF_Tactile-Fabric\2025\Fabric_time\Codes\Regression+time+TactNet_alldatatype_FBbranch\log';
time_list = {'manual','manual_L','manual_R', 'multi', 'multi_L','multi_R', 'crop_2s','crop_4s', 'crop_8s'};
model_list = {'TacNet'};
mode_list = {'time','branch','hybrid'};
act_list = {'identity', 'scaled_sigmoid', 'scaled_tanh'};

%% time
clearvars -except Working_dir time_list model_list mode_list act_list
fb_n = 3;
mode_idx = 0;
for iter_idx = 0:24
    load([Working_dir, '\Attn_TacNet' ...
                                '_' num2str(iter_idx) '_2'...
                                '_' mode_list{mode_idx+1} '_scaled_sigmoid'...
                                '_fb' num2str(fb_n) '_TokenImportance.mat'])
    temp_tattn(iter_idx+1,:) = imp_rollout;
    % plot(imp_rollout)
    % hold on
end
X = temp_tattn;
[E, T] = size(X);
if ~exist('t','var') || isempty(t), t = 1:T; end
m = mean(X, 1, 'omitnan');
s = std(X, 0, 1, 'omitnan')./sqrt(sum(~isnan(X),1));
lo = m - s;                               
hi = m + s;

figure; hold on;
fill([t, fliplr(t)], [lo, fliplr(hi)], 'k', ...
     'FaceAlpha', 0.15, 'EdgeColor','none');   % 검정 음영(연하게)
plot(t, m, 'k', 'LineWidth', 3);               % 검정 평균선
xlabel('Time (ms)'); ylabel('Attention Weight'); grid on;
legend({'\pm1 SD','Mean'}, 'Location','best');
axis tight
xticks(10:10:70)
xticklabels([10:10:70]*0.0256)

%% time_ver2
clearvars -except Working_dir time_list model_list mode_list act_list
fb_n = 3;
mode_idx = 0;
for iter_idx = 0:24
    load([Working_dir, '\Attn_TacNet' ...
                                '_' num2str(iter_idx) '_2'...
                                '_' mode_list{mode_idx+1} '_scaled_sigmoid'...
                                '_fb' num2str(fb_n) '_TokenImportance.mat'])
    temp_tattn(iter_idx+1,:) = imp_rollout;
    % plot(imp_rollout)
    % hold on
end

X = temp_tattn;
[E, T] = size(X);
if ~exist('t','var') || isempty(t), t = 1:T; end
m = mean(X, 1, 'omitnan');
s = std(X, 0, 1, 'omitnan')./sqrt(sum(~isnan(X),1));
lo = m - s;
hi = m + s;

figure; hold on;
fill([t, fliplr(t)], [lo, fliplr(hi)], 'k', ...
     'FaceAlpha', 0.15, 'EdgeColor','none');   % 검정 음영(연하게)
plot(t, m, 'k', 'LineWidth', 3);               % 검정 평균선
xlabel('Time (ms)'); ylabel('Attention Weight'); grid on;
legend({'\pm1 SD','Mean'}, 'Location','best');
axis tight
xticks(10:10:70)
xticklabels([10:10:70]*0.0256)

%% branch
clearvars -except Working_dir time_list model_list mode_list act_list
fb_n = 3;
mode_idx = 1;
for iter_idx = 0:24
    load([Working_dir, '\Attn_TacNet' ...
                                '_' num2str(iter_idx) '_2'...
                                '_' mode_list{mode_idx+1} '_scaled_sigmoid'...
                                '_fb' num2str(fb_n) '_TokenImportance.mat'])
    temp_battn(iter_idx+1,:) = imp_rollout;
    % plot(imp_rollout)
    % hold on
end

X = temp_battn;
[E, T] = size(X);
if ~exist('t','var') || isempty(t), t = 1:T; end
m = mean(X, 1, 'omitnan');
s = std(X, 0, 1, 'omitnan')./sqrt(sum(~isnan(X),1));
lo = m - s;                               
hi = m + s;

figure; hold on;
fill([t, fliplr(t)], [lo, fliplr(hi)], 'k', ...
     'FaceAlpha', 0.15, 'EdgeColor','none');   % 검정 음영(연하게)
plot(t, m, 'k', 'LineWidth', 3);               % 검정 평균선
xlabel('Branch'); ylabel('Attention Weight'); grid on;
legend({'\pm1 SD','Mean'}, 'Location','best');
axis tight
xticks(1:4)
xticklabels({"SA1", "SA2", "RA1", "RA2"})

%% hybrid time
clearvars -except Working_dir time_list model_list mode_list act_list
fb_n = 3;
mode_idx = 2;
for iter_idx = 0:24
    load([Working_dir, '\Attn_TacNet' ...
                                '_' num2str(iter_idx) '_2'...
                                '_' mode_list{mode_idx+1} '_scaled_sigmoid'...
                                '_fb' num2str(fb_n) '_TokenImportance.mat'])
    temp_hattn(iter_idx+1,:) = imp_rollout;
    % plot(imp_rollout)
    % hold on
end

X = temp_hattn(:,1:79);
[E, T] = size(X);
if ~exist('t','var') || isempty(t), t = 1:T; end
m = mean(X, 1, 'omitnan');
s = std(X, 0, 1, 'omitnan')./sqrt(sum(~isnan(X),1));
lo = m - s;                               
hi = m + s;

figure; hold on;
fill([t, fliplr(t)], [lo, fliplr(hi)], 'k', ...
     'FaceAlpha', 0.15, 'EdgeColor','none');
plot(t, m, 'k', 'LineWidth', 3);
xlabel('Time (ms)'); ylabel('Attention Weight'); grid on;
legend({'\pm1 SD','Mean'}, 'Location','best');
axis tight
xticks(10:10:70)
xticklabels([10:10:70]*0.0256)

%% hybrid branch
clearvars -except Working_dir time_list model_list mode_list act_list
fb_n = 3;
mode_idx = 2;
for iter_idx = 0:24
    load([Working_dir, '\Attn_TacNet' ...
                                '_' num2str(iter_idx) '_2'...
                                '_' mode_list{mode_idx+1} '_scaled_sigmoid'...
                                '_fb' num2str(fb_n) '_TokenImportance.mat'])
    temp_hattn(iter_idx+1,:) = imp_rollout;
    % plot(imp_rollout)
    % hold on
end
% mean_hattn = mean(temp_hattn,1);
% plot(mean_hattn(:,79:end),'LineWidth',3)
% axis tight
% xticks(1:1:4)
% xticklabels({'SA1','SA2','RA1','RA2'})
% xlabel('Branch')
% ylabel('Attention Weight')

X = temp_hattn(:,79:end);
[E, T] = size(X);
if ~exist('t','var') || isempty(t), t = 1:T; end
m = mean(X, 1, 'omitnan');
s = std(X, 0, 1, 'omitnan')./sqrt(sum(~isnan(X),1));
lo = m - s;                               
hi = m + s;

figure; hold on;
fill([t, fliplr(t)], [lo, fliplr(hi)], 'k', ...
     'FaceAlpha', 0.15, 'EdgeColor','none');   % 검정 음영(연하게)
plot(t, m, 'k', 'LineWidth', 3);               % 검정 평균선
xlabel('Branch'); ylabel('Attention Weight'); grid on;
legend({'\pm1 SD','Mean'}, 'Location','best');
axis tight
xticks(1:4)
xticklabels({"SA1", "SA2", "RA1", "RA2"})

%% hybrid all
clearvars -except Working_dir time_list model_list mode_list act_list
fb_n = 3;
mode_idx = 2;
for iter_idx = 0:24
    load([Working_dir, '\Attn_TacNet' ...
                                '_' num2str(iter_idx) '_2'...
                                '_' mode_list{mode_idx+1} '_scaled_sigmoid'...
                                '_fb' num2str(fb_n) '_TokenImportance.mat'])
    temp_hattn(iter_idx+1,:) = imp_rollout;
    % plot(imp_rollout)
    % hold on
end

X = temp_hattn;
[E, T] = size(X);
if ~exist('t','var') || isempty(t), t = 1:T; end
m = mean(X, 1, 'omitnan');
s = std(X, 0, 1, 'omitnan')./sqrt(sum(~isnan(X),1));
lo = m - s;                               
hi = m + s;

figure; hold on;
fill([t, fliplr(t)], [lo, fliplr(hi)], 'k', ...
     'FaceAlpha', 0.15, 'EdgeColor','none');   % 검정 음영(연하게)
plot(t, m, 'k', 'LineWidth', 3);               % 검정 평균선
xlabel('Time (ms)'); ylabel('Attention Weight'); grid on;
legend({'\pm1 SD','Mean'}, 'Location','best');
axis tight

line([79 79],[0, max(X,[],'all')])
xticks([10:10:70,79:82])
xticklabels([[10:10:70]*0.0256, {'SA1','SA2','RA1','RA2'}])
