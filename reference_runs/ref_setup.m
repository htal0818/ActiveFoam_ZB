function ref_setup(seed)
% Paths, compatibility shims and explicit seeding for Octave and MATLAB.
% shims/rng.m is a no-op so that ts_initialConf's rng('shuffle') cannot reseed.
% shims/heaviside.m and normrnd.m stand in for the Symbolic / Statistics toolboxes.
here=fileparts(mfilename('fullpath'));
addpath(fullfile(here,'shims'));
warning('off','all');
if exist('OCTAVE_VERSION','builtin')
    rand('seed',seed); randn('seed',seed);
else
    RandStream.setGlobalStream(RandStream('mt19937ar','Seed',seed));
end
end
