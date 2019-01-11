function output=FIR(A,W)
    [a,len]=size(A);
    output=zeros(a,len-W);
    for i=W:len
        output(:,i-W+1)=sum(A(:,i-W+1:i),2);
    end
    output=output/W;
end