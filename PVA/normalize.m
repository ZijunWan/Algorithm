function output = normalize(A)
    [a,b]=size(A);
    output=zeros(a,b);
    for i=1:a
        maxnum=max(abs(A(i,:)));
        output(i,:)=A(i,:)/maxnum;
    end
end
