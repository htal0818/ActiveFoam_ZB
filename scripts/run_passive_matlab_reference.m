function run_passive_matlab_reference(adhesion, seed, out, stepBudget)
% Record stress and restart states from the author's ts_simStepZero protocol.
if nargin < 4
    stepBudget = Inf;
end
if ~isfolder(out)
    mkdir(out);
end
config = struct('adhesion', adhesion, 'density', 1., 'cells', 100, ...
                'seed', seed, 'strain', 1.5, 'dt', .005, 'tauT', 10.);
checkpoint = fullfile(out, 'state.mat');
if isfile(checkpoint)
    saved = load(checkpoint, 'state');
    state = saved.state;
    assert(isequal(state.config, config), 'The checkpoint has different simulation parameters.');
    rng(state.rngState);
    fprintf('Resuming W/T0=%.2f at phase %d, iteration %d.\n', adhesion, state.phase, state.itc);
else
    fprintf('Initializing 100 cells at W/T0=%.2f, seed %d.\n', adhesion, seed);
    rng(seed, 'twister');
    state = initialize(config);
    writeCheckpoint(state, out);
end

stepped = 0;
while state.phase < 5 && stepped < stepBudget
    limit = [config.tauT, 6*config.tauT + 8*config.dt, 2*config.tauT, 30*config.tauT];
    if state.time > limit(state.phase)
        state = nextPhase(state);
        writeCheckpoint(state, out);
        continue
    end
    state = advance(state);
    stepped = stepped + 1;
    if state.phase >= 3 && mod(state.itc, 10) == 3
        state.nSamples = state.nSamples + 1;
        state.trace(state.nSamples,:) = [state.time, state.gmp.sstn, shearStress(state)];
    end
    if mod(state.itc, 500) == 0
        writeCheckpoint(state, out);
    end
end
writeCheckpoint(state, out);
end

function state = initialize(config)
bs = sqrt(config.cells);
psi = sqrt(config.density);
gmp = struct('tgAr', 1, 'sstn', 0, 'bs', bs, 'edpc', .15, ...
             'nFa', config.cells, 'lsc', psi, 'dt', config.dt, 'shEd', .01*2*sqrt(pi));
mcp = struct('beta', 0, 'gam', config.adhesion, 'del', 10, ...
             'psi', psi, 'mu', 0, 'taut', config.tauT);
[vrtx,edge,face,rg] = ts_initialConf(bs);
edge = resetTensions(edge, face, rg, mcp);
for vId = 1:size(vrtx,1)
    [vrtx,edge,face,gmp] = ts_t2ReverseTransition(vrtx,edge,face,rg,gmp,mcp,vId);
end
mcp.mu = .5/2 * (2-mcp.gam);
state = struct('config', config, 'phase', 1, 'itc', 0, 'time', 0., ...
               'shearTime', NaN, 'nSamples', 0, 'trace', zeros(6000,3), ...
               'vrtx', vrtx, 'edge', {edge}, 'face', {face}, ...
               'rg', rg, 'gmp', gmp, 'mcp', mcp);
end

function edge = resetTensions(edge, face, rg, mcp)
for eId = 1:size(edge{1},1)
    edge{1}(eId,rg.ei(3)) = ts_edgeFixedTension(face,edge,rg,mcp,eId);
end
end

function state = nextPhase(state)
switch state.phase
    case 1
        state.mcp.mu = 0;
        state.edge = resetTensions(state.edge,state.face,state.rg,state.mcp);
    case 2
        state.time = 0.;
        state.itc = 0;
    case 3
        state.shearTime = state.time;
        strain = state.config.strain;
        state.gmp.sstn = strain;
        xcol = state.rg.vi(3);
        state.vrtx(:,xcol) = state.vrtx(:,xcol) + strain*state.vrtx(:,xcol+1);
        for eId = 1:numel(state.edge{2})
            points = state.edge{2}{eId};
            points(:,1) = points(:,1) + strain*points(:,2);
            state.edge{2}{eId} = points;
        end
end
state.phase = state.phase + 1;
end

function state = advance(state)
vrtx = state.vrtx;
edge = state.edge;
face = state.face;
rg = state.rg;
gmp = state.gmp;
mcp = state.mcp;
gmp.eLnc = ts_edgeLenAll(edge,gmp);
[vrtx,edge,face] = ts_iteration(vrtx,edge,face,rg,gmp,mcp);
state.itc = state.itc + 1;
state.time = state.time + gmp.dt;
if mod(state.itc,3) == 0
    [vrtx,edge,face,gmp] = ts_topTransitionType1(vrtx,edge,face,rg,gmp,mcp);
end
switch mod(state.itc,5)
    case 1
        [vrtx,edge,face,gmp] = ts_topTransitionType3(vrtx,edge,face,rg,gmp,mcp);
    case 2
        for vId = 1:size(vrtx,1)
            [vrtx,edge,face] = ts_t4AdjacentTransition(vrtx,edge,face,rg,gmp,mcp,vId);
        end
    case 3
        [vrtx,edge,face,gmp] = ts_topTransitionType2(vrtx,edge,face,rg,gmp,mcp);
end
state.vrtx = vrtx;
state.edge = edge;
state.face = face;
state.gmp = gmp;
end

function value = shearStress(state)
value = 0.;
for eId = 1:numel(state.edge{2})
    delta = diff(state.edge{2}{eId},1,1);
    lengths = sqrt(sum(delta.^2,2));
    keep = lengths > 1e-12;
    tension = max(0., state.edge{1}(eId,state.rg.ei(3)));
    value = value + tension*sum(delta(keep,1).*delta(keep,2)./lengths(keep));
end
value = value * sqrt(state.config.density) / state.gmp.bs^2;
end

function writeCheckpoint(state, out)
state.rngState = rng;
temporary = fullfile(out, 'state.tmp.mat');
save(temporary, 'state', '-v7');
movefile(temporary, fullfile(out, 'state.mat'), 'f');
if state.nSamples > 0
    rows = state.trace(1:state.nSamples,:);
    origin = state.shearTime;
    if isnan(origin)
        origin = 2*state.config.tauT;
    end
    rows(:,1) = rows(:,1) - origin;
    trace = array2table(rows, 'VariableNames', {'time_tauR','strain','stress_over_sigma0'});
    temporary = fullfile(out, 'trace.tmp.csv');
    writetable(trace, temporary);
    movefile(temporary, fullfile(out, 'trace.csv'), 'f');
end
fprintf('W/T0=%.2f phase=%d step=%d time=%.4f samples=%d\n', ...
        state.config.adhesion, state.phase, state.itc, state.time, state.nSamples);
end
