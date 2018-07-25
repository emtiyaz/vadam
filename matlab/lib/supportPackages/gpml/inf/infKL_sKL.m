function [post nlZ dnlZ] = infKL_sKL(hyp, mean, cov, lik, x, y)
% SVI using adaptive gradient
% Wu Lin

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

lik_name = func2str(lik{1});

snu2=hyp.snu2;
n=size(x,1);

% GP prior
K = feval(cov{:}, hyp.cov, x);                  % evaluate the covariance matrix
m = feval(mean{:}, hyp.mean, x);                      % evaluate the mean vector
K=snu2*eye(n)+K;

%the size of mini batch= n_batch * mini_batch_rate
mini_batch_size=hyp.mini_batch_size;
assert (mini_batch_size>0)
mini_batch_num=ceil(n/mini_batch_size);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%init value
m_u=hyp.init_m;
C_u = chol(hyp.init_V,'lower');
C_u = C_u-diag(diag(C_u))+diag(log(diag(C_u)));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

assert(isfield(hyp,'learning_rate'))

%algorithms={'adadelta','momentum','rmsprop','adagrad','smorms3'};
if isfield(hyp,'opt_alg')
	switch hyp.opt_alg
	case 'momentum'
		momentum_m_u=zeros(n,1);
		momentum_C_u=zeros(n,n);
	case 'adadelta'
		assert(isfield(hyp,'epsilon'))
		assert(isfield(hyp,'decay_factor'))
		g_acc_m_u=zeros(n,1);
		g_delta_acc_m_u=zeros(n,1);
		g_acc_C_u=zeros(n,n);
		g_delta_acc_C_u=zeros(n,n);
	case 'rmsprop'
		assert(isfield(hyp,'epsilon'))
		assert(isfield(hyp,'decay_factor'))
		g_acc_m_u=zeros(n,1);
		g_acc_C_u=zeros(n,n);
        case 'adam'
                assert(isfield(hyp,'epsilon'))
                assert(isfield(hyp,'decay_factor_var'))
                assert(isfield(hyp,'decay_factor_mean'))
                g_mean_m_u=zeros(n,1);
                g_mean_C_u=zeros(n,n);
                g_var_m_u=zeros(n,1);
                g_var_C_u=zeros(n,n);
	case 'adagrad'
		assert(isfield(hyp,'epsilon'))
		g_acc_m_u=zeros(n,1);
		g_acc_C_u=zeros(n,n);
	case 'smorms3'
		assert(isfield(hyp,'epsilon'))
		g_acc_m_u=zeros(n,1);
		g_acc_square_m_u=zeros(n,1);
		mem_m_u=ones(n,1);
		g_acc_C_u=zeros(n,n);
		g_acc_square_C_u=zeros(n,n);
		mem_C_u=ones(n,n);
	otherwise
		error('do not support')
	end

end
assert(~isfield(hyp,'momentum'))
assert(~isfield(hyp,'adadelta'))
iter = 0;
pass=0;
max_pass=hyp.max_pass;
while pass<max_pass
	index=randperm(n);
	offset=0;
	mini_batch_counter=0;
	pass=pass+1;
	while mini_batch_counter<mini_batch_num
		iter = iter + 1;
		mini_batch_counter=mini_batch_counter+1;

		%mini batch
		tmp_idx = mini_batch_counter*mini_batch_size;
		idx=index( (tmp_idx-mini_batch_size+1):min(tmp_idx,n) );

		rate=hyp.learning_rate;
		weight=double(n)/size(x,1);

		alpha=K\(m_u-m);
		post_m=m_u;
		C = C_u-diag(diag(C_u))+diag(exp(diag(C_u)));
		post_v=sum(C'.*C',1)';

		if hyp.stochastic_approx==1
			[ll, gf, gv] = sampling_E(y(idx), post_m(idx), post_v(idx), lik, hyp.sample_size, hyp.lik);
		else
			switch lik_name
			case {'laplace','likLaplace','poisson','bernoulli_logit','likLogistic'}
				[ll, gf, gv] = E_log_p(lik_name, y(idx), post_m(idx), post_v(idx), hyp.lik);
			otherwise	 
				[ll,gf,g2f,gv] = likKL(post_v(idx), lik, hyp.lik, y(idx), post_m(idx));
			end
		end

		%mapping the change in a mini_batch to the change in the whole batch 
		df=zeros(n,1);
		df(idx)=weight*gf;
		dv=zeros(n,1);
		dv(idx)=weight*gv;

		g_rate=rate/(iter).^(hyp.power);
		g_m_u=alpha-df;

		g_C_u=tril(K\C - diag(2.0.*dv)*C);
		g_C_u=g_C_u-diag(diag(g_C_u))+diag(diag(g_C_u).*diag(C))+diag(-1.*ones(n,1));

		if isfield(hyp,'opt_alg')
			switch hyp.opt_alg
			case 'momentum'
				momentum_m_u=hyp.momentum .* momentum_m_u-g_rate .*g_m_u;
				m_u=m_u+momentum_m_u;
				momentum_C_u=hyp.momentum .* momentum_C_u-g_rate .*g_C_u;
				C_u=C_u+momentum_C_u;

			case 'adadelta'
				decay_factor=hyp.decay_factor;
				epsilon=hyp.epsilon;
				learning_rate=hyp.learning_rate;

				[g_m_u,g_acc_m_u,g_delta_acc_m_u] = adadelta_update(g_m_u,g_acc_m_u,g_delta_acc_m_u,decay_factor,epsilon,learning_rate);
				m_u=m_u-g_m_u;

				[g_C_u,g_acc_C_u,g_delta_acc_C_u] = adadelta_update(g_C_u,g_acc_C_u,g_delta_acc_C_u,decay_factor,epsilon,learning_rate);
				C_u=C_u-g_C_u;

			case 'rmsprop'
				decay_factor=hyp.decay_factor;
				epsilon=hyp.epsilon;
				learning_rate=hyp.learning_rate;

				[g_m_u,g_acc_m_u] = rmsprop_update(g_m_u,g_acc_m_u,decay_factor,epsilon,learning_rate);
				m_u=m_u-g_m_u;

				[g_C_u,g_acc_C_u] = rmsprop_update(g_C_u,g_acc_C_u,decay_factor,epsilon,learning_rate);
				C_u=C_u-g_C_u;

			case 'adagrad'
				epsilon=hyp.epsilon;
				learning_rate=hyp.learning_rate;

				[g_m_u,g_acc_m_u] = adagrad_update(g_m_u,g_acc_m_u,epsilon,learning_rate);
				m_u=m_u-g_m_u;

				[g_C_u,g_acc_C_u] = adagrad_update(g_C_u,g_acc_C_u,epsilon,learning_rate);
				C_u=C_u-g_C_u;

                        case 'adam'
                                decay_factor_var=hyp.decay_factor_var;
                                decay_factor_mean=hyp.decay_factor_mean;
                                epsilon=hyp.epsilon;
                                learning_rate=hyp.learning_rate;

                                [g_m_u,g_mean_m_u,g_var_m_u] = adam_update(g_m_u,g_mean_m_u,g_var_m_u,decay_factor_mean,decay_factor_var,epsilon,learning_rate,iter);
                                m_u=m_u-g_m_u;

                                [g_C_u,g_mean_C_u,g_var_C_u] = adam_update(g_C_u,g_mean_C_u,g_var_C_u,decay_factor_mean,decay_factor_var,epsilon,learning_rate,iter);
                                C_u=C_u-g_C_u;

			case 'smorms3'
				epsilon=hyp.epsilon;
				learning_rate=hyp.learning_rate;

				[g_m_u,g_acc_m_u,g_acc_square_m_u,mem_m_u] = smorms3_update(g_m_u,g_acc_m_u,g_acc_square_m_u,mem_m_u, epsilon,learning_rate);
				m_u=m_u-g_m_u;

				[g_C_u,g_acc_C_u,g_acc_square_C_u,mem_C_u] = smorms3_update(g_C_u,g_acc_C_u,g_acc_square_C_u,mem_C_u,epsilon,learning_rate);
				C_u=C_u-g_C_u;

			otherwise
				error('do not support')
			end
		else
			%sgd
			m_u=m_u-g_rate.*g_m_u;
			C_u=C_u-g_rate.*g_C_u;
		end

		if isfield(hyp,'save_iter') && hyp.save_iter==1
			global cache_nlz_iter
			global cache_iter

			post_m=m_u;
			alpha=K\(m_u-m);
			C=C_u-diag(diag(C_u))+diag(exp(diag(C_u)));
			post_v=sum(C'.*C',1)';
			switch lik_name
			case {'laplace','likLaplace','poisson','bernoulli_logit','likLogistic'}
				[ll_iter, df, dv] = E_log_p(lik_name, y, post_m, post_v, hyp.lik);
			otherwise	 
				[ll_iter,df,d2f,dv] = likKL(post_v, lik, hyp.lik, y, post_m);
			end
			W=-2.0*dv;
			sW=sqrt(abs(W)).*sign(W);
			nlZ2=compute_nlz(lik, hyp, sW, K, m, alpha, post_m, y);

			cache_iter=[cache_iter; iter];
			cache_nlz_iter=[cache_nlz_iter; nlZ2];
		end
	end

	%display nlz
	post_m=m_u;
	alpha=K\(m_u-m);
	C=C_u-diag(diag(C_u))+diag(exp(diag(C_u)));
	post_v=sum(C'.*C',1)';
	switch lik_name
	case {'laplace','likLaplace','poisson','bernoulli_logit','likLogistic'}
		[ll_iter, df, dv] = E_log_p(lik_name, y, post_m, post_v, hyp.lik);
	otherwise	 
		[ll_iter,df,d2f,dv] = likKL(post_v, lik, hyp.lik, y, post_m);
	end
	W=-2.0*dv;
	sW=sqrt(abs(W)).*sign(W);
	nlZ=compute_nlz(lik, hyp, sW, K, m, alpha, post_m, y);
	fprintf('pass:%d) %.4f\n', pass, nlZ);

	if hyp.is_save==1
		global cache_post;
		global cache_nlz;

		L = chol(eye(n)+sW*sW'.*K);
		post.sW = sW;                                           
		post.alpha = alpha;
		post.L = L;                                    

		cache_post=[cache_post; post];
		cache_nlz=[cache_nlz; nlZ];
	end
end
L = chol(eye(n)+sW*sW'.*K); %L = chol(sW*K*sW + eye(n)); 
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

function [grad,g_mean,g_var] = adam_update(gradient,g_mean,g_var,decay_factor_mean,decay_factor_var,epsilon,learning_rate,times)
        g_mean=decay_factor_mean .* g_mean + (1.0-decay_factor_mean) .* (gradient);
        g_var=decay_factor_var .* g_var + (1.0-decay_factor_var) .* (gradient.^2);
        g_mean_hat=g_mean ./ (1.0-(decay_factor_mean.^times));
        g_var_hat=g_var ./ (1.0-(decay_factor_var.^times));
        grad=learning_rate .* g_mean_hat ./ (sqrt(g_var_hat)+epsilon);
        %learning_rate_adjusted=learning_rate.*sqrt(1.0-(decay_factor_var.^times))./(1.0-(decay_factor_mean.^times));
        %grad=learning_rate_adjusted .* g_mean ./ (sqrt(g_var)+epsilon);
end


function [grad,g_acc,g_delta_acc] = adadelta_update(gradient,g_acc,g_delta_acc,decay_factor,epsilon,learning_rate)
	g_acc=decay_factor .* g_acc + (1.0-decay_factor) .* (gradient.^2);
	grad= (learning_rate .* gradient .* sqrt(g_delta_acc + epsilon) ./ sqrt(g_acc+epsilon) );
	g_delta_acc=decay_factor .* g_delta_acc + (1.0-decay_factor) .* (grad.^2);
end

function [grad,g_acc] = rmsprop_update(gradient,g_acc,decay_factor,epsilon,learning_rate)
	g_acc=decay_factor .* g_acc + (1.0-decay_factor) .* (gradient.^2);
	grad=learning_rate .* gradient ./ sqrt(g_acc+epsilon);
end

function [grad,g_acc] = adagrad_update(gradient,g_acc,epsilon,learning_rate)
	g_acc=g_acc + (gradient.^2);
	grad=learning_rate .* gradient ./ sqrt(g_acc+epsilon);
end

function [grad,g_acc,g_acc_square,mem] = smorms3_update(gradient,g_acc,g_acc_square,mem,epsilon,learning_rate)
	r=1.0./(mem+1.0);
	g_acc=(1.0-r) .* g_acc + r .* gradient;
	g_acc_square=(1.0-r) .* g_acc_square + r .* (gradient.^2);

	tmp=(g_acc.^2) ./ (g_acc_square+epsilon);
	grad=gradient.* min(learning_rate, tmp) ./ (sqrt(g_acc_square) + epsilon);
	mem=1.0 + mem .* (1.0 - tmp); 
end
