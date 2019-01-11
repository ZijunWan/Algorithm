load('move.mat');
load('spike.mat');
binlen=0.05;
firingrate=spike/binlen;
firingrate_train=firingrate(:,1:end-1000);
firingrate_test=firingrate(:,end-1001:end);
move=move';
move_train=move(:,1:end-1000);
move_test=move(:,end-1001:end);

output=func(move_test,move_train,firingrate_train,firingrate_test,5);
plot(output);
