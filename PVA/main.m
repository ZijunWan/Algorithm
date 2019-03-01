clear all;
clc;
load('F:\dataset\MoveData\move.mat');
load('F:\dataset\MoveData\spike.mat');
binlen=0.1;
W=5;
nn=size(spike,1);
d=size(move,1); 
firingrate=spike/binlen;
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
%判断spike中有没有发放率一直是0的，这种在求偏好角度和调制深度的时候会出现问题，因为需要求矩阵的逆运算
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
legend("decode","real");
title(['cc= ',num2str(cc(1,2)), '     ', 'RMSE= ', num2str(RMSE)]);



function b=train(pos,spk)
    [d,len]=size(pos);
    pos=[ones(1,len);pos];
    if(len~=length(spk))
        error('length of pos and spk is not the same');
    end
    nn=size(spk,1);
    X=pos*pos';
    b=[];
    for i=1:nn
        Y=pos*spk(i,:)';
        b=[b, pinv(X)*Y];
    end
end

