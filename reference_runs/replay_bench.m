function replay_bench(statefile,nsteps)
% Time nsteps of the reference loop from a saved state (same state on any machine).
% Paths to the author code (and optionally fast_override/) must already be set.
ref_setup(1);
s=load(statefile); [vrtx,edge,face,rg,gmp,mcp,itc]=deal(s.vrtx,s.edge,s.face,s.rg,s.gmp,s.mcp,s.itc);
t0=tic;
for k=1:nsteps, [vrtx,edge,face,gmp,itc]=ref_sched(vrtx,edge,face,rg,gmp,mcp,itc); end
t=toc(t0); [sxy,phi]=ref_stress(edge,face,rg,gmp,mcp);
fprintf('%s: %d steps in %.1f s = %.2f steps/s; V=%d E=%d sxy=%.6f phi=%.5f (using %s)\n', ...
    statefile,nsteps,t,nsteps/t,size(vrtx,1),size(edge{1},1),sxy,phi,which('ts_edgeRangeCheck'));
end
