% @Author: amishkin
% @Date:   2018-07-10T14:22:48-07:00
% @Email:  amishkin@cs.ubc.ca
% @Last modified by:   aaronmishkin
% @Last modified time: 2018-07-20T18:29:37-07:00


function [nlz, log_loss, Sigma, mu] = my_methods(method_name, y, X, gamma, y_te, X_te, options, mu_start, sigma_start)

    Sigma_all = [];
    mu_all = [];
    fprintf('%s\n',method_name);
    [N,D] = size(X);

    % set default options
    [max_iters, lowerBoundTol, display, num_samples, beta_start, alpha_start, decay_rate, mini_batch_size] = myProcessOptions(options, ...
    'max_iters', 2000, 'lowerBoundTol',1e-4, 'display', 1, 'num_samples', 1, 'beta', 0.1, 'alpha', 0.8, 'decay_rate', 0.55, 'mini_batch_size', N);

    % because we use LogisticLoss
    y_recoded = 2*y-1;

    mu = mu_start;
    Sigma = sigma_start;
    S = inv(Sigma);

    % print log_loss at iter = 0
    post_dist.mean = mu;
    post_dist.covMat = Sigma;
    iter = 0;

    [pred, log_lik]=get_loss(iter, post_dist, X, y, gamma, X_te, y_te);
    log_loss(iter+1) = pred.log_loss;
    nlz(iter+1) = -log_lik;

    beta_momentum = 0.1;
    grad_momentum =  zeros(D,1);

    % iterate
    for iter = 1:max_iters

        % Decay the learning rates.
        if decay_rate > 0
            alpha = alpha_start / (1 + iter^(decay_rate));
            beta = beta_start / (1 + iter^(decay_rate));
        else
            alpha = alpha_start;
            beta = beta_start;
        end

        % select a minibatch
        if mini_batch_size < N
            idx = unidrnd(N,[mini_batch_size,1]);
            Xi = X(idx, :);
            yi = y_recoded(idx, :); % use recoded to -1,1
        else  % batch mode, no minibatch exist
            Xi = X;
            yi = y_recoded; % use recoded to -1,1
        end

        weight = N/mini_batch_size;

        U = chol(Sigma);
        % compute gradients
        g = 0; H = 0; g_prev = 0;
        for i = 1:num_samples
            sample = randn(D,1);
            w = mu + U'*sample;
            [~, gi, ~] = LogisticLoss(w, Xi, yi);
            [~, ~, Hi] = LogisticLossGN(w, Xi, yi);
            g = g + gi;
            H = H + Hi;
        end

        g = (g./num_samples)*weight;
        g_prev = (g_prev./num_samples)*weight;
        H = (H./num_samples)*weight;

        switch method_name
        case 'Vadam'
            Hhat = diag(diag((g * g')  ./ N));

            grad_momentum = ((1-beta_momentum) .* grad_momentum) + ((g + gamma .* mu) .* beta_momentum);
            grad_momentum_corrected = grad_momentum ./ (1 - (1 - beta_momentum)^iter);

            S = (1-beta) * S + beta * Hhat;
            S_corrected = S ./ (1 - (1 - beta)^iter);
            Sigma = inv(S_corrected + (eye(D) .* gamma));
            prod = (sqrt(S_corrected) + (eye(D) .* gamma)) \ grad_momentum_corrected;

            mu = mu - alpha.*prod;
        case 'VOGN'
            grad_momentum = ((1-beta_momentum) .* grad_momentum) + ((g + gamma .* mu) .* beta_momentum);
            grad_momentum_corrected = grad_momentum ./ (1 - (1 - beta_momentum)^iter);

            % update (not efficiently implemented, but simple)
            S = inv(Sigma);
            S = (1-beta)*S + beta*(diag(diag(H)) + diag(gamma));
            Sigma = inv(S);

            mu = mu - alpha*(Sigma*(grad_momentum_corrected));
        otherwise
            error('There is no method of that name!')
        end
    end

    post_dist.mean = mu(:);
    post_dist.covMat = Sigma;
    [pred, log_lik] = get_loss(iter, post_dist, X, y, gamma, X_te, y_te);
    log_loss = pred.log_loss;
    nlz = -log_lik;
end
