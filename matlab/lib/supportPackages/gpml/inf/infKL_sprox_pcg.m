function [post nlZ dnlZ] = infKL_sprox_pcg(hyp, mean, cov, lik, x, y)
% PG-SVI

if hyp.is_cached==1
	global cache_post;
	global cache_nlz;
	global cache_idx;
	
	post=cache_post(cache_idx);
	nlZ=cache_nlz(cache_idx);
	if nargout>2
		warning('to be implemented\n');
		dnlZ = NaN;
	end
	return 
end

snu2=hyp.snu2;
n=size(x,1);
nu=round(sqrt(n))+1;
rot180   = @(A)   rot90(rot90(A));                     % little helper functions
chol_inv = @(A) rot180(chol(rot180(A))')\eye(nu);                 % chol(inv(A))

% GP prior
K = feval(cov{:}, hyp.cov, x);                  % evaluate the covariance matrix
m = feval(mean{:}, hyp.mean, x);                      % evaluate the mean vector
K=snu2*eye(n)+K;

lik_name = func2str(lik{1});
mini_batch_size=hyp.mini_batch_size;
assert (mini_batch_size>0)
mini_batch_num=ceil(n/mini_batch_size);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%init value
post_m=hyp.init_m;%k=0
tW = zeros(n,1);%k=-1
post_v=diag(hyp.init_V);%k=0
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

iter = 0;%iteration
pass=0;%pass
max_pass=hyp.max_pass;
beta = hyp.learning_rate;
r = 1/(beta+1);
index=1:n;
assert (mini_batch_size==1)
diagK=diag(K);

xu = x(1:nu,:);
Kuu = feval(cov{:}, hyp.cov, xu, xu); 
Ku = feval(cov{:}, hyp.cov, xu, x); 
R0 = chol_inv(Kuu+snu2*eye(nu));  V = R0*Ku;
%d0 = zeros(n,1); %Nystrom
d0 = diagK-sum(V.*V,1)'; %FITC
B=zeros(n,n);

while pass<max_pass
	index=randperm(n);
	offset=0;
	mini_batch_counter=1;
	pass=pass+1;

	tic
	idx=index(1);
	post_m_single=post_m(idx);
	post_v_single=post_v(idx);
	while mini_batch_counter<=mini_batch_num
		iter=iter+1;
		weight=double(n)/size(x(idx,:),1);

		if hyp.stochastic_approx==1
			[ll, gf, gv] = sampling_E(y(idx), post_m_single, post_v_single, lik, hyp.sample_size, hyp.lik);
		else
			switch lik_name
			case {'laplace','likLaplace','poisson','bernoulli_logit','likLogistic'}
				[ll, gf, gv] = E_log_p(lik_name, y(idx), post_m_single, post_v_single, hyp.lik);
			otherwise	 
				[ll,gf,d2f,gv] = likKL(post_v_single, lik, hyp.lik, y(idx), post_m_single);
			end
		end

		% pseudo observation
		pseudo_y = m + K(:,idx)*(weight*gf) - post_m;

		tW = r.*tW;
		tW(idx) = tW(idx)+(1-r).*((-2*weight)*gv);%tW^{k}, where W=-2*gv
		sW = sqrt(abs(tW)) .* sign(tW);
		A = eye(n)+sW*sW'.*K;


		%using FITC/Nystrom as pre-conditioner
		nu=size(Kuu,1);
		n=size(Ku,2);
		d_inv = 1.0 ./ ( ones(n,1) + sW.*sW.*d0 );
		tmp1 = repmat( (sW.*d_inv)',nu,1).*Ku; %Ku*sW*diag(d_inv)
		tmp2 = (repmat( (sW.*sW.*d_inv)',nu,1).*Ku)*Ku';
		tmp3=tmp2+Kuu;


		%L = chol(A); %L = chol(sW*K*sW + eye(n)); 
		b = sW.*pseudo_y;
		res_m=my_pcg(A,b,d_inv,tmp1,tmp3);
		post_m = post_m + (1-r).*(pseudo_y - K*(sW.*res_m) );%post_m = post_m + (1-r).*(pseudo_y - K*(sW.*(L\(L'\(sW.*pseudo_y)))));

		if mini_batch_counter>=mini_batch_num
			break
		end

		mini_batch_counter=mini_batch_counter+1;
		idx=index( mini_batch_counter );
		%T = L'\(sW.*K(:,idx)); %T  = L'\(sW*K);
		b = sW.*K(:,idx);
		res_v=my_pcg(A,b,d_inv,tmp1,tmp3);
		post_v_single = diagK(idx) - b'*res_v;%post_v_single = diagK(idx) - sum(T.*T,1)'; % v = diag(inv(inv(K)+diag(W))); %v^{k+1}
		assert (post_v_single > 0)
		post_m_single=post_m(idx);

		if isfield(hyp,'save_iter') && hyp.save_iter==1
			global cache_nlz_iter
			global cache_iter

			alpha=K\(post_m-m);
			nlZ2=compute_nlz(lik, hyp, sW, K, m, alpha, post_m, y);
			cache_iter=[cache_iter; iter];
			cache_nlz_iter=[cache_nlz_iter; nlZ2];
		end
	end
	toc

	alpha=K\(post_m-m);
	sW = sqrt(abs(tW)) .* sign(tW);
	L = chol(eye(n)+sW*sW'.*K); %L = chol(sW*K*sW + eye(n)); 
	nlZ=compute_nlz(lik, hyp, sW, K, m, alpha, post_m, y);

	if isfield(hyp,'save_iter') && hyp.save_iter==1
		if pass==1
			global num_iters_at_pass;
			num_iters_at_pass=iter;
		end
		fprintf('pass:%d) at %d iter %.4f %.4f\n', pass, iter, nlZ, nlZ2);
	else
		fprintf('pass:%d) %.4f\n', pass, nlZ);
	end

	if hyp.is_save==1
		global cache_post;
		global cache_nlz;

		post.sW = sW;                                             % return argument
		post.alpha = alpha;
		post.L = L;      

		cache_post=[cache_post; post];
		cache_nlz=[cache_nlz; nlZ];
	end
end

alpha=K\(post_m-m);
post.sW = sW;                                             % return argument
post.alpha = alpha;
post.L = L;                                              % L'*L=B=eye(n)+sW*K*sW

nlZ=compute_nlz(lik, hyp, sW, K, m, alpha, post_m, y);
fprintf('final: %.4f\n', nlZ);

if nargout>2
  warning('to be implemented\n');
  dnlZ = NaN;
end
end

function [x it]=my_pcg(K,b,d_inv,tmp1,tmp3)
	n=size(b,1);
	x = zeros(n,1);
	%r = b - K*x; %if x is non-zero
	r = b;
	z = (d_inv).*r - tmp1'*(tmp3\(tmp1*r));%z = P\r;, where P=eye(n)+diag(sW)*( Ku'*(Kuu\Ku)+diag(d0) )*diag(sW)
	p = z;
	it=0;
	while it<n
		alpha = r'*z / (p'*(K*p));
		x = x + alpha*p;
		r_prev = r;
		r = r - alpha*(K*p);
		if sum(abs(r))<n*1e-2
			break
		end
	z_prev = z;
	z = (d_inv).*r - tmp1'*(tmp3\(tmp1*r));%z = P\r;
	beta = z'*r/(z_prev'*r_prev);
	p = z + beta*p;
	it=it+1;
	end
	if it>sqrt(n)*3
		fprintf('Warning! >sqrt(n) %d iterations n=%d\n',it,n)
	end
end

%function [res]=low_rank_inv(d_inv, tmp1, tmp3, x)
	%res = (d_inv).*x - tmp1'*(tmp3\(tmp1*x)); 
%end

%function [res]=low_rank_inv(d0,Ku,Kuu,sW,x)
        %nu=size(Kuu,1);
        %n=size(Ku,2);
        %d_inv = 1.0 ./ ( ones(n,1) + sW.*sW.*d0 );
        %tmp1 = repmat( (sW.*d_inv)',nu,1).*Ku; %Ku*sW*diag(d_inv)
        %tmp2 = (repmat( (sW.*sW.*d_inv)',nu,1).*Ku)*Ku';
        %res = (d_inv).*x - tmp1'*((tmp2+Kuu)\(tmp1*x)); 

	%%A=eye(n)+diag(sW)*( Ku'*(Kuu\Ku)+diag(d0) )*diag(sW);
	%%res2=A\x;
	%%fprintf('low:%.6f %.6f\n',res(3),res2(3));
%end
