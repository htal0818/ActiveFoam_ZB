function [ar,cen]=ts_faceArea(fEd,eMd,bs,sstn)
% Batched 2-by-2 pivoted LU determinants, accumulated in polygon order.
fCrd=ts_faceVrtxCrd(fEd,eMd,bs,sstn);
cen=mean(fCrd);
next=fCrd([2:end,1],:);
swap=abs(fCrd(:,1))<abs(next(:,1));
top=fCrd;
bottom=next;
top(swap,:)=next(swap,:);
bottom(swap,:)=fCrd(swap,:);
determinant=zeros(size(swap));
keep=top(:,1)~=0;
schur=bottom(keep,2)-(bottom(keep,1)./top(keep,1)).*top(keep,2);
determinant(keep)=top(keep,1).*schur;
determinant(swap)=-determinant(swap);
ar=0;
for i=1:numel(determinant)
    ar=ar+determinant(i)/2;
end
end
