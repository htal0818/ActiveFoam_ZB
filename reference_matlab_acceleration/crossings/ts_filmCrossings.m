function [ecr,ept,crps1,crps2] = ts_filmCrossings(edge,eId,gmp,skip)
% Evaluate the same line-intersection arithmetic across all segment pairs.
first = edge{2}{eId(1)};
second = edge{2}{eId(2)};
split = size(first,1);
points = ts_crdLocal([first;second],gmp.bs,gmp.sstn);
first = points(1:split,:);
second = points(split+1:end,:);
a = first(1:end-1,:);
b = second(1:end-1,:);
da = diff(first,1,1);
db = diff(second,1,1);
dx = b(:,1).' - a(:,1);
dy = b(:,2).' - a(:,2);
determinant = da(:,1).*db(:,2).' - da(:,2).*db(:,1).';
mm = (dx.*da(:,2) - dy.*da(:,1))./determinant;
kk = (dx.*db(:,2).' - dy.*db(:,1).')./determinant;
hit = determinant ~= 0 & mm > 0 & mm < 1 & kk > 0 & kk < 1;
if ~isempty(skip)
    hit(skip(1),skip(2)) = false;
end
ecr = nnz(hit);
if ecr == 0
    ept = [];
else
    [i,j] = find(hit);
    fraction = kk(sub2ind(size(hit),i,j));
    ept = a(i,:) + fraction.*da(i,:);
end
if nargout > 2
    [crps2,crps1] = find(hit.');
    crps1 = crps1(:);
    crps2 = crps2(:);
end
end
