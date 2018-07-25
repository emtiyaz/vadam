function [f, gm, gv] = sampling_E(y, m, v, lik, sample_size, varargin)
% This function approximates E( log p(y|x) ) where 
% expectation is wrt p(x) = N(x|m,v) with mean m and variance v.
% params are optional parameters required for approximation

  % vectorize all variables
  y = y(:); m = m(:); v = v(:);
  n=length(y);
  sv=sqrt(v);
  points=normrnd(0,1,1,sample_size);

  lik_str = lik{1}; if ~ischar(lik_str), lik_str = func2str(lik_str); end
  switch lik_str
  case 'likLogistic'
	  yf=(sv * points + repmat(m,1,sample_size)) .* repmat(y,1,sample_size);
	  lp=yf; ok = -35<yf; lp(ok)=-log(1+exp(-yf(ok))); % log of likelihood
  case 'likGauss'
	  hyp_lik=varargin{1};
	  sn2 = exp(2*hyp_lik);
	  yf=(sv * points + repmat(m,1,sample_size)) - repmat(y,1,sample_size);
	  lp = -(yf).^2./sn2/2-log(2*pi*sn2)/2;
  otherwise
	error('do not support');
  end

  f= sum(lp,2)./sample_size;
  gm=(sum(repmat(points,n,1) .* lp, 2) ./sample_size) ./ sv;
  part1=-0.5 .* f ./ v;
  part2=0.5 .* (sum(repmat(points.^2,n,1) .* lp, 2) ./sample_size)  ./ v;
  gv=part1+part2;
