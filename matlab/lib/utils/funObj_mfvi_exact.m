function [f,g] = funObj_mfvi_exact(v, y, X, lambda)
% Primal objective function (Omega = inv(Sigma)) for mean-field
% f = 0.5*(logdet(V) - logdet(Sigma) - tr(V*SigmaInv) - (m-mu)'*SigmaInv*(m-mu)
%     + L) - sum_d fb(mbar_d, vbar_d)
% where mbar = X*m, vbar = diag(X*V*X')
%
% Written by Emtiyaz,
% Modified by Wu Lin
  [D L] = size(X);

  Omega = diag(lambda);
  %Extract mean, Cholesky and bias
  m = v(1:L);
  sigma = v(L+1:end);
  U = diag(sigma);

  % compute V
  V = diag(sigma.^2);

  % compute kl and its gradient
  kl = 0.5*(2*sum(log(diag(U))) + sum(log(lambda(2:end))) - trace(V*Omega) -m'*(Omega*m) + L);

  % contribution from the bound
  mbar = X*m;
  vbar = sum(X.*(V*X')',2); % diag(X*V*X') efficient
  [fb, gmb, gvb] = E_log_p('bernoulli_logit', y, mbar, vbar, []);
  fb = -fb;
  gm_lvb = X'*(-gmb);
  gV_lvb = X'*bsxfun(@times, -gvb, X); %efficient X'*diag(gllp/2)*X;

  % final
  f = kl - sum(fb);
  gm = - Omega*m - gm_lvb;
  gU = diag(1./diag(U)) - diag(U*Omega) - diag(2*U*gV_lvb);

  g=[gm(:); diag(gU)];

  % return
  g=-g;
  f=-f;

