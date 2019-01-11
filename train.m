% 2 demension population vector parameter generation
% pos:dimension+1 * timelength with bias=1 in the 1 row
% spk: neuronnum * timelength
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