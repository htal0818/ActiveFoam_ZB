function ref_stepzero_stress(bs,gamma,psi,ststn,seed,outfile)
% ts_simStepZero.m with the state dumps replaced by a stress trace every 10 steps.
% Stage lengths, mu values, tension resets and the strain step are copied from it.
ref_setup(seed);
[beta,delta,taut,edpc]=deal(0,10,10,0.15);
[gmp,mcp]=deal(struct);
[gmp.tgAr,gmp.sstn,gmp.bs,gmp.edpc]=deal(1,0,bs,edpc);
[mcp.beta,mcp.gam,mcp.del,mcp.psi,mcp.mu,mcp.taut]=deal(beta,gamma,delta,psi,0,taut);
[gmp.nFa,gmp.lsc,gmp.dt]=deal(gmp.bs^2,mcp.psi*sqrt(gmp.tgAr),0.005);
gmp.shEd=0.01*2*sqrt(pi);
t0=tic; fid=fopen(outfile,'w'); ffprintf(fid,'stage,t,sxy,phi,nvrtx\n');
[vrtx,edge,face,rg]=ts_initialConf(gmp.bs);
for ii=1:size(edge{1},1), edge{1}(ii,rg.ei(3))=ts_edgeFixedTension(face,edge,rg,mcp,ii); end
for ii=1:size(vrtx,1)
    [vrtx,edge,face,gmp]=ts_t2ReverseTransition(vrtx,edge,face,rg,gmp,mcp,ii);
end
fprintf('init done %.0fs\n',toc(t0)); ref_flush();
[itc,ttMx,ttTm]=deal(0,mcp.taut,0);
mcp.mu=0.5/2*(2-mcp.gam);
while (ttTm<=ttMx)
    [vrtx,edge,face,gmp,itc]=ref_sched(vrtx,edge,face,rg,gmp,mcp,itc); ttTm=ttTm+gmp.dt;
end
fprintf('anneal done %.0fs (%d steps)\n',toc(t0),itc); ref_flush();
mcp.mu=0;
for ii=1:size(edge{1},1), edge{1}(ii,rg.ei(3))=ts_edgeFixedTension(face,edge,rg,mcp,ii); end
while (ttTm<=6*ttMx+8*gmp.dt)
    [vrtx,edge,face,gmp,itc]=ref_sched(vrtx,edge,face,rg,gmp,mcp,itc); ttTm=ttTm+gmp.dt;
    if mod(itc,2000)==0, fprintf('quench step %d %.0fs\n',itc,toc(t0)); ref_flush(); end
end
fprintf('quench done %.0fs (%d steps)\n',toc(t0),itc); ref_flush();
[itc,ttMx,ttTm]=deal(0,30*mcp.taut,0);
while (ttTm<=2*mcp.taut)
    [vrtx,edge,face,gmp,itc]=ref_sched(vrtx,edge,face,rg,gmp,mcp,itc); ttTm=ttTm+gmp.dt;
    if mod(itc,10)==3
        [s,p]=ref_stress(edge,face,rg,gmp,mcp); ffprintf(fid,'pre,%.3f,%.6f,%.5f,%d\n',ttTm-2*mcp.taut,s,p,size(vrtx,1)); 
    end
end
gmp.sstn=ststn;
vrtx(:,rg.vi(3))=vrtx(:,rg.vi(3))+ststn*vrtx(:,rg.vi(3)+1);
for edc=1:size(edge{2},1)
    emd=edge{2}{edc}; emd(:,1)=emd(:,1)+ststn*emd(:,2); edge{2}{edc}=emd;
end
[s,p]=ref_stress(edge,face,rg,gmp,mcp); ffprintf(fid,'jump,%.3f,%.6f,%.5f,%d\n',0,s,p,size(vrtx,1)); 
fprintf('strain applied %.0fs, sxy=%.4f\n',toc(t0),s); ref_flush();
while (ttTm<=ttMx)
    [vrtx,edge,face,gmp,itc]=ref_sched(vrtx,edge,face,rg,gmp,mcp,itc); ttTm=ttTm+gmp.dt;
    if mod(itc,10)==3
        [s,p]=ref_stress(edge,face,rg,gmp,mcp); ffprintf(fid,'post,%.3f,%.6f,%.5f,%d\n',ttTm-2*mcp.taut,s,p,size(vrtx,1)); 
    end
    if mod(itc,4000)==0, fprintf('relax step %d %.0fs\n',itc,toc(t0)); ref_flush(); end
end
fclose(fid); fprintf('finished %.0fs\n',toc(t0));
end
