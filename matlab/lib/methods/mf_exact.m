% @Author: amishkin
% @Date:   2018-07-10T14:56:18-07:00
% @Email:  amishkin@cs.ubc.ca
% @Last modified by:   aaronmishkin
% @Last modified time: 2018-07-20T18:30:05-07:00

function [nlz, log_loss, Sigma, mu] = mf_exact(method_name, y, X, gamma, y_te, X_te, options, mu_start, sigma_start)

    fprintf('%s\n',method_name);
    [N,D] = size(X);

    % set default options
    [max_iters, lowerBoundTol, display, num_samples, beta_start, alpha_start, decay_rate, mini_batch_size] = myProcessOptions(options, ...
    'max_iters', 2000, 'lowerBoundTol',1e-4, 'display', 1, 'num_samples', 1, 'beta', 0.1, 'alpha', 0.8, 'decay_rate', 0.55, 'mini_batch_size', N);


    % minfunc options
    optMinFunc = struct('display', display,...
        'Method', 'lbfgs',...
        'DerivativeCheck', 'off',...
        'LS', 2,...
        'recordPath', 1, ...
        'recordPathIters', 1, ...
        'MaxIter', max_iters+1,...
        'MaxFunEvals', max_iters+1,...
        'TolFun', lowerBoundTol,......
        'TolX', lowerBoundTol);

    V = sigma_start;
    m = mu_start;

    v0 = [m; sqrt(diag(V))];
    funObj = @funObj_mfvi_exact;

    % compute loss at iter =0
    post_dist.mean = m;
    post_dist.covMat =V;
    iter = 0;
    [pred, log_lik]=get_loss(iter, post_dist, X, y, gamma, X_te, y_te);
    nlz(iter+1)=-log_lik;%nlz0
    log_loss(iter+1)=pred.log_loss;%log_loss0

    % optimize using minfunc
    [v, f, exitflag, inform] = minFunc(funObj, v0, optMinFunc, y, X, gamma);
    v_all = inform.trace.x;

    % compute loss for iter>0
    for ii=1:size(v_all,2)
        vi = v_all(:,ii);
        post_dist.mean = vi(1:D);
        U = diag(vi(D+1:end));
        post_dist.covMat = U'*U;
        if sum(eig(post_dist.covMat) <= 1e-8) > 0
            ALERT = 'NOT PD'
            nlz(ii) = NaN;
            log_loss(ii) = NaN;
            continue
        end
    end
    [pred, log_lik] = get_loss(ii-1, post_dist, X, y, gamma, X_te, y_te);
    nlz =-log_lik;
    log_loss = pred.log_loss;

    Sigma = post_dist.covMat;
    mu = post_dist.mean;
end
