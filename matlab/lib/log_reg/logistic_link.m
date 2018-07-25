function [g, H, b] = logistic_link(y,X,xMean,cholP,priorMeanTimePrec,priorPrec,S)
%Written by Wu Lin
[g, H, b] = logistic_link_no_prior(y,X,xMean,cholP,S);
g = g + priorMeanTimePrec - priorPrec*b; % prior
H = H -priorPrec; %prior
end

function [g, H, b] = logistic_link_no_prior(y,X,xMean,cholP,S)
%Written by Wu Lin
  y = y>0;
  y = 2*y-1;%n by 1
  %GPML uses -1 and 1 encoding

  [n k]= size(X);

  % sample from q
  fn = normrnd(0, 1, [k, S]);
  betaDraws=bsxfun(@plus, xMean, cholP\fn); %k by S

  fn = (X*betaDraws)';%S by n
  y = y(:);

  hyp.lik = [];
  lik = {@likLogistic};

  % compute MC approximation (code taken from GPML)
  y = repmat(y(:)', S, 1);
  [f, df, d2f] = feval(lik{:}, hyp, y, fn, [], 'infLaplace');
  f = mean(f,1)';
  gm = mean(df,1)';
  gv = mean(d2f,1)'/2;
  g = X'*gm;

  %assert( size(gv, 2) == 1 )
  tmp1 = bsxfun(@times,X,2*gv);%diag(2*gv)*X
  H = X'*tmp1;
  b = mean(betaDraws,2);
end
