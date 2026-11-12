function [vrtx,edge,face,gmp,itc]=ref_sched(vrtx,edge,face,rg,gmp,mcp,itc)
% one pass of the loop body shared by every stage of ts_simStepZero.m (verbatim order)
gmp.eLnc=ts_edgeLenAll(edge,gmp);
[vrtx,edge,face]=ts_iteration(vrtx,edge,face,rg,gmp,mcp);
itc=itc+1;
if mod(itc,3)==0
    [vrtx,edge,face,gmp]=ts_topTransitionType1(vrtx,edge,face,rg,gmp,mcp);
end
if mod(itc,5)==1
    [vrtx,edge,face,gmp]=ts_topTransitionType3(vrtx,edge,face,rg,gmp,mcp);
end
if mod(itc,5)==2
    for ii=1:size(vrtx,1)
        [vrtx,edge,face]=ts_t4AdjacentTransition(vrtx,edge,face,rg,gmp,mcp,ii);
    end
end
if mod(itc,5)==3
    [vrtx,edge,face,gmp]=ts_topTransitionType2(vrtx,edge,face,rg,gmp,mcp);
end
end
