clear all;
clc;
load('move.mat');
load('spike.mat');
move=normalize(move);
move_train=move(:,1:end-1000);
move_test=move(:,end-999:end);
spike_train=spike(:,1:end-1000);
spike_test=spike(:,end-999:end);
move_test=[ones(1,1000);move_test];
b=train(move_train,spike_train);
result=PVA(b,move_test,spike_test);
result=normalize(result);
cc= corrcoef(result, move_test(2:3,:));
RMSE=sqrt((result-move_test(2:3,:))*(result-move_test(2:3,:))')/500;
plot(result(1,:));
 hold on;
% figure;
plot(move_test(2,:));
title(['cc=  ',num2str(cc(1,2)),'     ','RMSE=  ',num2str(RMSE(1,2))])

