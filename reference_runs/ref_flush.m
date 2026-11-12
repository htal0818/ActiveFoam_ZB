function ref_flush()
if exist('OCTAVE_VERSION','builtin'), fflush(stdout); else, drawnow('update'); end
end
