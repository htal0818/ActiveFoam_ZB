function [sxy,phi,pres]=ref_stress(edge,face,rg,gmp,mcp)
% Paper Eq. 4/5 (not in the supplied code): tension part of sigma_xy per unit area,
% sum over curved-edge segments of max(T,0)*lx*ly/|l|, divided by the box area.
s=0;
for e=1:size(edge{1},1)
    T=max(edge{1}(e,rg.ei(3)),0);
    if T>0
        d=diff(edge{2}{e},1,1); L=sqrt(sum(d.^2,2)); ok=L>1e-12;
        s=s+T*sum(d(ok,1).*d(ok,2)./L(ok));
    end
end
sxy=s/gmp.bs^2;
phi=sum(face{1}(:,rg.fi(2)))/gmp.bs^2;
pres=0;
end
