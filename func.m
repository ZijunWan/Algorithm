function output=func(speed_test,speed_train,firingrate_train,firingrate_test,W)

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
end



len=length(trainX);
for i=1:len
    if(trainY(2,i)<0)
        svm_train(i,1)=-1;
        svm_train(i,2:end)=trainY(:,i);
    elseif(trianY(2,i)>0)
        svm_train(i,1)=1;
        svm_train(i,2:end)=trainY(:,i);
    else
        svm_train(i,:)=0; 
    end
end

for i=1:len
    if(svm_train(i,1)==0)
        svm_train(i,:)=[];
        i=i-1;
    end
end
