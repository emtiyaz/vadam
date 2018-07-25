function [nlzs log_losses times_cache] = GaussVarApproxFactorGradientGLM(compute_loss,priorMeanTimesPrec,priorPrec,features,nrSteps,nSamples,stepSize,X,y,X_te,y_te,init_a1,init_a2)
%Code is adapted from https://github.com/TimSalimans/LinRegVB/blob/master/probit/GaussVarApproxFactorGradientGLM.m

nlzs=zeros(nrSteps+1,1);
log_losses=zeros(nrSteps+1,1);
times_cache=zeros(nrSteps+1,1);

% dimension
[n,k] = size(features);% features is X

% initialize results to zero
XX12 = zeros(n,1);
XX22 = -ones(n,1);
XY1 = zeros(n,1);
XY2 = zeros(n,1);

dm1 = init_a1*ones(n,1);
dm2 = init_a2*ones(n,1);
gamma = diag(priorPrec);
assert( all(gamma>0) )

w=dm2;
XX22 = -w./dm2;
XY2 = -w.*ones(n,1);
XX12 = 0.*ones(n,1);
XY1 = dm1-w.*XX12./XX22;

N = n;
assert( all(gamma>0) )
sW0 = 1 ./ sqrt(gamma);
half_XX=repmat(sW0,1,N).*(X');
if k>n
    XX = half_XX'*half_XX;%N by N % X*diag(gamma)*X'
end
vv0 = sum(half_XX.*half_XX, 1);
vv0 = vv0';%diag( X*diag(1./gamma)*X')

y0 = y>0;
%check = unique(y0);
%assert( length(check) ==2 )
%assert( check(1) ==0 )
%assert( check(2) ==1 )

% do stochastic approximation
for i=1:nrSteps
    tic;
    % do regression
    natpar_meanTimesPrec = (XX22.*XY1 - XX12.*XY2)./XX22;%tm1
    natpar_prec = XY2./XX22;%tm2

    if k>n
        %For "big p and small n" problem
        %we use the matrix inversion lemma
        %this section is written by Wu Lin
        sW = sqrt(abs(natpar_prec)).*sign(natpar_prec);
        L = chol(eye(N)+sW*sW'.*XX);
        v=L'\(repmat(sW,1,N).*XX);
        zVar = vv0-sum(v.*v,1)';
        zMean = XX*natpar_meanTimesPrec - v'*v*natpar_meanTimesPrec;
    else
        %For "big n and small p" problem
        % get posterior for beta
        cholP = chol(features'*(features.*repmat(natpar_prec,1,k)) + priorPrec);
        meanTimesPrec = features'*natpar_meanTimesPrec + priorMeanTimesPrec;
        % get posterior for z=features*beta
        zMean = features*(cholP\(cholP'\meanTimesPrec));
        zVar = sum((features/cholP).^2,2);
    end
    tt1 = toc;

    if compute_loss==1
        natpar_meanTimesPrec0 = (XX22.*XY1 - XX12.*XY2)./XX22;
        natpar_prec0 = XY2./XX22;
        post_prec = features'*(features.*repmat(natpar_prec0,1,k)) + priorPrec;
        post_mean = post_prec\(features'*natpar_meanTimesPrec0 + priorMeanTimesPrec);
        post_cov = post_prec\eye(k);
        post_dist.mean = post_mean;
        post_dist.covMat = post_cov;
        [pred, log_lik]=get_loss(i-1, post_dist, X, y, gamma, X_te, y_te);
        nlzs(i) = -log_lik;
        log_losses(i) = pred.log_loss;
    end

    tic;
    [XXs12, XXs22, XYs1, XYs2] = get_samples(y0,zMean,zVar,n,nSamples);

    % update old statistics
    XX12 = (1-stepSize)*XX12 + stepSize*XXs12;
    XX22 = (1-stepSize)*XX22 + stepSize*XXs22;
    XY1 = (1-stepSize)*XY1 + stepSize*XYs1;
    XY2 = (1-stepSize)*XY2 + stepSize*XYs2;

    tt2 = toc;

    times_cache(i+1) = tt2 + tt1;
end
times_cache = cumsum(times_cache);

if compute_loss==1
    % output final results
    natpar_meanTimesPrec = (XX22.*XY1 - XX12.*XY2)./XX22;
    natpar_prec = XY2./XX22;
    approxPrec = features'*(features.*repmat(natpar_prec,1,k)) + priorPrec;
    approxMean = approxPrec\(features'*natpar_meanTimesPrec + priorMeanTimesPrec);
    i = nrSteps+1;
    approxV = approxPrec\eye(k);
    post_dist.mean = approxMean;
    post_dist.covMat = approxV;
    [pred, log_lik]=get_loss(i-1, post_dist, X, y, gamma, X_te, y_te);
    nlzs(i) = -log_lik;
    log_losses(i) = pred.log_loss;
end

end

function [XXs12, XXs22, XYs1, XYs2] = get_samples(y,zMean,zVar,n,S)
%Modified by Wu Lin
% sample - we use antithetics, this simplifies the inverses
rn = randn(n,S);
r = [rn -rn];

stdv = sqrt(zVar);
zDraws = repmat(zMean,1,2*S) + repmat(stdv,1,2*S).*r;

% get gradient of log likelihood values
llhGrad = likLogistic_grad(y,zDraws);

XXs12 = -zMean;
XXs22 = -mean(repmat(stdv,1,S).*rn.^2,2);
XYs1 = mean(llhGrad,2);
XYs2 = mean(r.*llhGrad,2);
end


function [grad] = likLogistic_grad(y,f)
%Written by Wu Lin
y = y>0;
%check = unique(y);
%assert( length(check) ==2 )
%assert( check(1) ==0 )
%assert( check(2) ==1 )
p = f.*repmat(2*y-1,1,size(f,2));
ps = max(0, p);
r  = exp(-ps) ./ ( exp(p-ps) + exp(-ps) );                    % r = 1/(1+exp(p))
grad = repmat(2*y-1,1,size(f,2)).*r;
end
