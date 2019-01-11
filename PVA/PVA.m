%2 demension population vector test
%b=dimension+1 * neuronnum
%pos=dimension+1 * length with bias=1 in the first row
%spk=neuronnum * length
function output=PVA(b,pos,spk)
    [d,nn]=size(b);
    if(d~=size(pos,1))
        error('dimension not the same');
    end
    if(nn~=size(spk,1))
        error('neuronnum not the same');
    end
    if(length(pos)~=length(spk))
        error('length not the same');
    end
    len=length(pos);
    pv=zeros(2,len);
    for i=1:len
        for j=1:nn
            pv(:,i)=pv(:,i)+(spk(j,i)-b(1,j))* [b(2,j);b(3,j)];
        end
    end
    output=pv/nn;
    
end

