function [ecr,ept,crps1,crps2] = ts_edgeCrossAdjacent(edge,rg,eId,gmp)
end1 = edge{1}(eId(1),rg.ei(1):rg.ef(1));
end2 = edge{1}(eId(2),rg.ei(1):rg.ef(1));
n1 = size(edge{2}{eId(1)},1)-1;
n2 = size(edge{2}{eId(2)},1)-1;
if end1(1) == end2(1)
    skip = [1,1];
elseif end1(1) == end2(2)
    skip = [1,n2];
elseif end1(2) == end2(1)
    skip = [n1,1];
else
    skip = [n1,n2];
end
[ecr,ept,crps1,crps2] = ts_filmCrossings(edge,eId,gmp,skip);
end
