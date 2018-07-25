function [f, gm, gv] = E_log_p_mc(v, lik, hyp, y, m, S)
% This function approximates E( log p(y|x) ) where
% expectation is wrt p(x) = N(x|m,v) with mean m and variance v.
% params are optional parameters required for approximation
% Written by Emtiyaz (EPFL)
% Modified on June 10, 2015
  y = y>0;
  y = 2*y-1;
  %GPML uses -1 and 1 encoding
  y = y(:); m = m(:); v = v(:);

  % sample from q
  n = length(y);
  s = sqrt(v);
  fn = normrnd(0, 1, [S, 1]);
  fn = bsxfun(@times, s(:)', fn(:));
  fn = bsxfun(@plus, m(:)', fn);

  % compute MC approximation (code is taken from GPML)
  y = repmat(y(:)', S, 1);
  [f, df, d2f] = feval(lik{:}, hyp, y, fn, [], 'infLaplace');
  f = mean(f,1)';
  gm = mean(df,1)';
  gv = mean(d2f,1)'/2;
end
