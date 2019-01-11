clear all;
clc;
load('move.mat');
load('spike.mat');
% load('G:\dataset\Decoding_Data\hc_data_raw.mat');
% load('G:\dataset\Decoding_Data\m1_data_raw.mat');
binlen=0.1;
W=5;
nn=size(spike,1);
d=size(move,1);
firingrate=spike/binlen;

firingrate=FIR(firingrate,W);

move=normalize(move);

speed=move(:,2:end)-move(:,1:end-1);
speed=speed(:,W-1:end);
speed_train=speed(:,1:end-999);
speed_test=speed(:,end-1000:end);
firingrate_train=firingrate(:,1:end-999);
firingrate_test=firingrate(:,end-1000:end);
move_test=move(:,end-1000-W:end-W);

len=length(speed_test);
b=train(speed_train,firingrate_train);
inposition=move(:,end-1000-W);
index=[];
for i=1:nn
    if(b(2,i)==0 && b(3,i)==0)
        index=[index,i];
    end
end

temp=0;
for i=1:length(index)
    b(:,index(i)-temp)=[];
    firingrate_test(index(i)-temp,:)=[];
    temp=temp+1;
end

nn=size(firingrate_test,1);
pd=zeros(2,nn);
for i=1:nn
    m(i)=sqrt(b(2,i)^2+b(3,i)^2);
    pd(:,i)=[b(2,i)/m(i) ; b(3,i)/m(i)];
end
u=zeros(2,len);
for i=1:len
    for j=1:nn
        r(i,j)=(firingrate_test(j,i)-b(1,j))/m(j);
        u(:,i)=u(:,i)+r(i,j)*pd(:,j)-0.005;
    end
    u(:,i)=(d/nn)*u(:,i);
end
p=zeros(2,len+1);
p(:,1)=inposition;
for i=2:len+1
    p(:,i)=p(:,i-1)+u(:,i-1)*binlen;
end
plot(p(1,2:end));
hold on;
plot(move_test(1,:));

cc=corrcoef(move_test(1,:),p(1,2:end));
RMSE=(move_test(1,:)-p(1,2:end))*(move_test(1,:)-p(1,2:end))'/len;












