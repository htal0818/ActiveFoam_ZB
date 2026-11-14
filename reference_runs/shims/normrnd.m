function r=normrnd(mu,sigma,varargin)
if nargin>2, r=mu+sigma.*randn(varargin{:}); else, r=mu+sigma.*randn(size(mu+sigma)); end
end
