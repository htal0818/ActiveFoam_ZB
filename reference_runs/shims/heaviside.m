function y=heaviside(x)
% MATLAB Symbolic Toolbox convention: heaviside(0)=1/2
y=double(x>0)+0.5*double(x==0);
end
