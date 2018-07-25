function [post nlZ dnlZ] = infKL_init(hyp, mean, cov, lik, x, y)

snu2=hyp.snu2;
% GP prior
n = size(x,1);
K = feval(cov{:}, hyp.cov, x);                  % evaluate the covariance matrix
m = feval(mean{:}, hyp.mean, x);                      % evaluate the mean vector
K=snu2*eye(n)+K;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%init value
post_m=hyp.init_m;%k=0
tW = zeros(n,1);%k=-1
post_v=diag(hyp.init_V);%k=0
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
lik_name = func2str(lik{1});

alpha=K\(post_m-m);
sW = sqrt(abs(tW)) .* sign(tW);
nlZ=compute_nlz(lik, hyp, sW, K, m, alpha, post_m, y);

post.sW = sW;                                             % return argument
post.alpha = alpha;
L = chol(eye(n)+sW*sW'.*K);
post.L = L;                                              % L'*L=B=eye(n)+sW*K*sW

fprintf('final: %.4f\n', nlZ);
if nargout>2
  warning('to be implemented\n');
  dnlZ = NaN;
end
