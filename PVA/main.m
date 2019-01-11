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

% firingrate=FIR(firingrate,W);

move=normalize(move);
firingrate=firingrate(:,1:end-1);
speed=move(:,2:end)-move(:,1:end-1);
speed_train=speed(:,W:end-999);
speed_test=speed(:,end-1000:end);

firingrate_train=firingrate(:,1:end-999);
firingrate_test=firingrate(:,end-1000:end);

move_test=move(:,end-1001:end-1);

firingrate_train=FIR(firingrate_train,W);

len=length(speed_test);
b=train(speed_train,firingrate_train);
inposition=move(:,end-1001);
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
        u(:,i)=u(:,i)+r(i,j)*pd(:,j);
    end
    u(:,i)=(d/nn)*u(:,i);
end
p=zeros(2,len+1);
p(:,1)=inposition;
for i=2:len+1
    p(:,i)=p(:,i-1)+u(:,i-1)*binlen;
end

cc=corrcoef(move_test(1,:),p(1,2:end));
RMSE=(move_test(1,:)-p(1,2:end))*(move_test(1,:)-p(1,2:end))'/len;

plot(p(1,2:end));
hold on;
plot(move_test(1,:));
title(['cc= ',num2str(cc(1,2)), '     ', 'RMSE= ', num2str(RMSE)]);

% move=move(1,:);
% speed=move(1,2:end)-move(1,1:end-1);
% spike=spike(:,1:end-1);
% speed_train=speed(1,1:end-1000);
% speed_test=speed(1,end-999:end);
% spike_train=spike(1,1:end-1000);
% spike_test=spike(1,end-999:end);
% move_test=move(1,end-998:end-1);
% len=length(speed_train);
% nn=size(spike_train,1);
% b=zeros(size(speed_train,1)+1,nn);
% temp3=sum(speed_train);
% temp4=speed_train*speed_train';
% temp5=(sum(speed_train))^2;
% for i=1:nn
%      temp1=sum(spike_train(i,:)*speed_train(1,:)');
%      temp2=sum(spike_train(i,:));
%      b(1,i)=(len*temp1-temp2*temp3)/(len*temp4-temp5);
%      b(2,i)=temp2/len-b(1,i)*temp3/len;
% end
% 
% temp=1;
% index=[];
% for i=1:nn
%         if(b(2,i)==0)
%                 index(temp)=i;
%                 temp=temp+1;
%         end
% end
% 
% if(index~=[])
% temp=1;
% for i=1:length(index)
%         b(1,index(tmep))=[];
%         spike_test(index(temp),:)=[];
%         temp=temp+1;
% end
% end
% lent=length(spike_test); 
% nn=size(spike_test,1);
% u=zeros(1,lent);
% for i=1:lent
%         for j=1:nn
%                 r(j,i)=(spike_test(j,i)-b(1,j))/b(2,j);
%                 u(i)=u(i)+r(j,i)*b(2,j);
%         end
%         u(i)=u(i)/nn;
% end
% p(1)=move(1,end-997)+u(1);
% for i=2:lent
%         p(i)=p(i-1)+u(i);
% end
% plot(p);
% hold on;
% plot(move_test);
















