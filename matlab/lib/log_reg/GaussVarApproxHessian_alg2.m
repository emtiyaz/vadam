function [nlzs log_losses times_cache] = GaussVarApproxHessian_alg2(compute_loss,guessMean,guessPrec,gradFun,nrSteps,nSamples,stepSize,X,y,X_te,y_te,init_a1,init_a2)
%Code is adapted from https://github.com/TimSalimans/LinRegVB/blob/master/probit/GaussVarApproxHessian.m

nlzs=zeros(nrSteps+1,1);
log_losses=zeros(nrSteps+1,1);
times_cache=zeros(nrSteps+1,1);

% dimension
k=length(guessMean);
N = size(X,1);

% initial statistics for stochastic approximation
dm1 = init_a1*ones(N,1);
dm2 = init_a2*ones(N,1);
tm11 = X'*dm1;
tmp1 = bsxfun(@times,X,dm2);%diag(dm2)*X
tm22 = X'*tmp1;

gamma = diag(guessPrec);
sW = 1 ./ sqrt( gamma );
L = chol(eye(k)+sW*sW'.*tm22);
half_V = L'\(diag(sW)); %V=half_V'*half_V; %%V= ( tm2+diag(gamma) )^{-1}
m = half_V'*(half_V*tm11);%note that prior_mean = 0
V = half_V'*half_V;

P = V\eye(k);
z = m;
a = zeros(k,1);
% do stochastic approximation
for i=1:nrSteps
    tic;
    % cholesky factor of current inverse variance matrix
    cholP = chol(P);
    % sample
    xMean = cholP\(cholP'\a) + z;

    post_mean = xMean;
    post_cov = cholP\(cholP'\eye(k));
    tt1 = toc;

    if compute_loss==1
        post_dist.mean = post_mean;
        post_dist.covMat = post_cov;
        [pred, log_lik]=get_loss(i-1, post_dist, X, y, gamma, X_te, y_te);
        nlzs(i) = -log_lik;
        log_losses(i) = pred.log_loss;
    end

    tic;
    [g H b] =gradFun(xMean, cholP, nSamples);

    % update statistics
    P = (1-stepSize)*P - stepSize*H;
    a = (1-stepSize)*a + stepSize*g;
    z = (1-stepSize)*z + stepSize*b;
    tt2 = toc;
    times_cache(i+1) = tt1 + tt2;
end
times_cache = cumsum(times_cache);

if compute_loss==1
    approxPrec = P;
    cholP = chol(approxPrec);
    approxMean = cholP\(cholP'\a) + z;
    approxV = cholP\(cholP'\eye(k));

    i = nrSteps+1;
    post_dist.mean = approxMean;
    post_dist.covMat = approxV;
    [pred, log_lik]=get_loss(i-1, post_dist, X, y, gamma, X_te, y_te);

    nlzs(i) = -log_lik;
    log_losses(i) = pred.log_loss;
end

end
