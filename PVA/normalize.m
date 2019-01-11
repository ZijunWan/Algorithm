function output = normalize(A)
    [a,b]=size(A);
    output=zeros(a,b);
    for i=1:a
        maxnum=max(abs(A(i,:)));
        minnum=min(A(i,:));
%         output(i,:)=(2*A(i,:)-minnum-maxnum)/(maxnum-minnum);
        output(i,:)=A(i,:)/maxnum;
    end
end
