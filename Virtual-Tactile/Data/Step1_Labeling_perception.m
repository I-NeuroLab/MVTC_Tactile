clear all
close all
clc

%% set parameter
% Tactile
width_list = {'200','300','500','700','1000','1500','2000'};
SR_list = {'1','2','3',};
H_list = {'100','300','500'};

%% load survey data
sublist = {'PKY','LHT_2','YDY','LSW','KNR','PTW','LHS','LES','KJY'};
sample_num = 66;
for subidx = 1:9
    S_data = readmatrix('D:\LHT\Survey_Results(72 samples).xlsx','Sheet',[sublist{subidx}]);
    S_label = S_data(3:9,2:10);
    Y = [];
    k = [];
    Cnt = 1;
    for S_idx = 1:3
        for H_idx = 2:3
            for W_idx = 1:7
                Y = cat(1,Y,[ones(sample_num,1)*Cnt, ones(sample_num,1)*(S_label(W_idx,3*(S_idx-1)+H_idx))]);
                S_label(W_idx,3*(S_idx-1)+H_idx)
                k = [k, S_label(W_idx,3*(S_idx-1)+H_idx)];
                Cnt = Cnt+1;
            end
        end
    end
    Y = Y-1;
    c1 = sum(k==1)/sample_num;
    c2 = sum(k==2)/sample_num;
    c3 = sum(k==3)/sample_num;
    c4 = sum(k==4)/sample_num;
    c5 = sum(k==5)/sample_num;
    save(['D:\LHT\New_synthtic\Data\Square\Y_' num2str(subidx) '_42.mat'],"Y");
end
