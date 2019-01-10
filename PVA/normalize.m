function output = normalize(A)
    [a,b]=size(A);
    for i=1:a
        rmax=max(A(i,:));
        rmin=min(A(i,:));
        output(i,:)=(A(i,:)-rmin)/rmax;
    end
end
