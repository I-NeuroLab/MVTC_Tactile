clear all
close all
clc

%% set parameter
% Tactile
width_list = {'200','300','500','700','1000','1500','2000'};
SR_list = {'1','2','3',};
H_list = {'100','300','500'};

%% load data
X1 = [];
X2 = [];
fs = 2500;
low_cutoff = 1;
high_cutoff = 400;
order = 4;
[b, a] = butter(order, [low_cutoff high_cutoff]/(fs/2), 'bandpass');

for W_idx = 5
    for S_idx = 1:3
        for H_idx = 1:3
            hz_true(3*(H_idx-1)+S_idx, W_idx) = 20000/str2double(width_list{W_idx})/(1+str2double(SR_list{S_idx}));
            [x, t, pos] = make_square_pattern_signal(str2num(width_list{W_idx}),...
                str2num(SR_list{S_idx}), H_idx, 0.2);
            [signal, t, pos, segments] = make_square_pattern_signal_scanpause_segment(str2num(width_list{W_idx}),...
                str2num(SR_list{S_idx}), H_idx, 0.1);

            X1 = [X1; segments];
            if H_idx ~= 1
                X2 = [X2; segments];
            end
            % 
            % for i = 1:size(segments,1)
            %     data_filt(i, :) = filtfilt(b, a, segments(i, :));
            % end
            % 
            % figure(2);
            % subplot(2,1,1);
            % plot(data_filt(1,:));
            % xlabel('Time (s)');
            % ylabel('Signal');
            % title('First segment (3초: 0.25초+2.5초+0.25초)');
            % 
            % subplot(2,1,2);
            % plot(data_filt(end,:));
            % xlabel('Time (s)');
            % ylabel('Signal');
            % title('Last segment (3초: 0.25초+2.5초+0.25초)');

            % subplot(6,10,trial_idx)
            figure(2)
            pwelch(segments(1,:),[],[],[],2500); xlim([0 0.06])
            clear segments
        end
    end
end
% X = X1;
% save('D:\LHT\New_synthtic\Data\Square\X_63.mat','X');
% clear X
% 
% X = X2;
% save('D:\LHT\New_synthtic\Data\Square\X_42.mat','X');
