function nlZ=compute_nlz(lik, hyp, sW, K, m, alpha, mu, y_batch)
% compute the negative lower bound
% Wu Lin
	n = size(mu,1);
	L = chol(eye(n)+sW*sW'.*K);                            % L'*L=B=eye(n)+sW*K*sW
	V = L'\(repmat(sW,1,n).*K);
	Sigma = K - V'*V;

	post_v = diag(Sigma);
	%indeed, post_m is mu
	post_m=K*alpha+m;

	lik_name = func2str(lik{1});
	switch lik_name
	case {'laplace','likLaplace','poisson','bernoulli_logit','likLogistic'}
		ll_iter=E_log_p(lik_name, y_batch, post_m, post_v, hyp.lik); 
	otherwise
		ll_iter= likKL(post_v, lik, hyp.lik, y_batch, post_m);
	end

	A = (eye(n)+K*diag(sW.^2))\eye(n);                           % A = Sigma*inv(K)
	nlZ = -sum(ll_iter) - (logdet(A)-alpha'*(mu-m)-trace(A)+n)/2;  % upper bound on -lZ


% log(det(A)) for det(A)>0 using the LU decomposition of A
function y = logdet(A)
  [L,U] = lu(A); u = diag(U); 
  if prod(sign(u))~=det(L), error('det(A)<=0'), end
  y = sum(log(abs(u)));
