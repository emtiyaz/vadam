function [pred, log_lik] = get_loss(iter, post_dist, X, y, gamma, X_te, y_te)

    % log likelihood
    val = [post_dist.mean(:); packcovariance(triu(chol(post_dist.covMat)))];
    f = chol_log_reg(val, y, X, gamma);
    log_lik = -f;

    pred = get_log_loss(X_te, y_te, post_dist);
    % [loss, p_hat] = mc_log_loss(y_te, X_te, post_dist);
    % pred.log_loss = loss;
    % pred.p_hat = p_hat;
    % print final value
    % fprintf('%d) objFun %.3f log_loss=%.3f\n',  iter, f, pred.log_loss);
end

function pred = get_log_loss(X_te, y_te, post_dist)
    % prediction
    % m_star = X*m, v_star = X*V*X'
    m_star = X_te*post_dist.mean;
    v_star = sum(X_te.*(X_te*post_dist.covMat),2);
    %v_star = diag(X_te*(post_dist.covMat*X_te'));

    % p_hat = \int sigmoid(x) N(x|m_star,v_star) dx
    p_hat = exp(likLogistic([],[], m_star(:), v_star(:)));
    % using sampling
    %S = 10000;
    %N_te = length(y_te);
    %z = bsxfun(@plus, m_star, bsxfun(@times, sqrt(v_star(:)), randn(N_te,S)));
    %p_hat = mean(sigmoid(z), 2);
    log_loss = compute_log_loss(y_te, p_hat);

    % output
    pred.p_hat = p_hat;
    pred.log_loss = log_loss;
end

function log_loss = compute_log_loss(y_te, p_hat)
    % log_loss = y.*log2(p) + (1-y).*log2(1-p)
    p_hat = max(eps,p_hat); p_hat = min(1-eps,p_hat);
    err = y_te.*log2(p_hat) + (1-y_te).*log2(1-p_hat);
    log_loss = -mean(err);
end

function [loss, p_hat] = mc_log_loss(y_te, X_te, post_dist)
    num_samples = 100;
    [N,D] = size(X_te);
    U = chol(post_dist.covMat);

    preds = zeros(N,1);

    for i = 1:num_samples
        theta = post_dist.mean + U' * randn(D,1);
        preds = preds + X_te * theta;
    end

    preds = preds ./ num_samples;
    p_hat = sigmoid(preds);
    loss = compute_log_loss(y_te, p_hat);
end
