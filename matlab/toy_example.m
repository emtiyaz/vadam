% @Author: aaronmishkin
% @Date:   2018-07-26T12:02:01-07:00
% @Email:  amishkin@cs.ubc.ca
% @Last modified by:   aaronmishkin
% @Last modified time: 2018-07-26T13:25:51-07:00
addpath(genpath('.'))

clear all;
% Set the variance of prior (alpha=1/lambda)
alpha=100;

% Generate the synthetic data.
dataset_name = 'murphy_synth';
[y, X, ytest, Xtest] = get_data_log_reg(dataset_name, 0);
t = (y+1) / 2;
D = 2;

% Limits and grid size for contour plotting
Range = 30;
Step=0.5;
[w1,w2]=meshgrid(-Range:Step:Range,-Range:Step:Range);
[n,n]=size(w1);
W=[reshape(w1,n*n,1) reshape(w2,n*n,1)];

%% prior, likelihood, posterior
f=W*X';
Log_Prior = log(gaussProb(W, zeros(1,D), eye(D).*alpha));
Log_Like = W*X'*t - sum(log(1+exp(f)),2);
Log_Joint = Log_Like + Log_Prior;
post = exp(Log_Joint - logsumexpPMTK(Log_Joint(:),1));

% Identify the MAP estimate
[i,j]=max(Log_Joint);
wmap = W(j,:);

% Compute the mf-exact solution.
D = 2;
sig = [1;1];
m = zeros(D,1);
v0 = [m; sig(:)];
funObj = @funObj_mfvi_exact;
optMinFunc = struct('display', 0, 'Method', 'lbfgs', 'DerivativeCheck', 'off','LS', 2, 'MaxIter', 100, 'MaxFunEvals', 100, 'TolFun', 1e-4, 'TolX', 1e-4);
gamma = ones(D,1)./alpha;
[v, f, exitflag, inform] = minFunc(funObj, v0, optMinFunc, t, X, gamma);
w_exact_vi = v(1:D);
U = v(D+1:end);
C_exact_vi = diag(U.^2);


% Run VOGN and Vadam
[N,D] = size(X);
maxIters = 500000;
colors = {'r','b','g','m'};

method = {'Vadam', 'VOGN'};
beta_momentum = 0.1;
grad_average = zeros(D,1);
w_all = zeros(2,maxIters,length(method));
Sigma_all = zeros(2,2,maxIters,length(method));

for m = 1:length(method)
   setSeed(1);
   name = method{m}
   % initialize
   w = [1, 1]';
   ss = 0.01;% alpha
   gamma = 0.99;% 1 - beta;

   beta_orig = 0.1
   ss_orig = 0.1
   for t = 1:maxIters
      %fprintf('%d) %.3f %.3f\n', t, w(1), w(2));
      M = 10;
      nSamples = 1;

      ss = ss_orig / (1 + t^(.55));
      beta = beta_orig / (1 + t^(.55));

      gamma = 1 - beta;
      switch name
      case 'VOGN'
          M = 1;
          if t == 1
              S_corrected = ones(D,1);
              S = S_corrected;
          end
      case 'Vadam'
          if t == 1;
            S_corrected = ones(D,1);
            S = S_corrected;
        end
      otherwise
          if t == 1; S = 100*eye(D); end
      end

      % draw minibatches
      i = unidrnd(N, [M,1]);

      % compute g and H
      g = 0; H = 0;

      Prec = S_corrected + eye(D)./alpha;
      Sigma = inv(Prec);
      U = chol(Sigma); % upper triangle matrix
      for k = 1:nSamples
         % for the k'th sample
         epsilon =  U' * randn(D,1);
         wt = w + epsilon;
         [~,gpos,Hpos] = LogisticLoss(wt,X(i,:),y(i));

         % gradient
         gk = (gpos / M);

         % Hessian
         wt = w + epsilon;
         [~,~,Gpos] = LogisticLossGN(wt,X(i,:),y(i,:));
         Hk = (Gpos / M);

         g = g + gk./nSamples;
         H = H + Hk./nSamples;
      end
      % unbiased estimate
      ghat = N*g;
      Hhat = N*diag(diag(H));

     switch name
     case 'Vadam'
        Hhat = diag(diag((ghat * ghat')  ./ N));

        grad_average = ((1-beta_momentum) .* grad_average) + ((ghat + (w ./ alpha)) .* beta_momentum);
        grad_average_corrected = grad_average ./ (1 - (1 - beta_momentum)^t);

        S = gamma * S + (1-gamma) * Hhat;
        S_corrected = S ./ (1 - (gamma)^t);
        w = w - ss*( (sqrt(S_corrected) + eye(D)./alpha) \ grad_average_corrected);

        w_all(:,t,m) = w(:);
        Sigma_all(:,:,t,m) = inv( S_corrected + eye(D)./alpha );
     case 'VOGN'
        grad_average = ((1-beta_momentum) .* grad_average) + ((ghat + (w ./ alpha)) .* beta_momentum);
        grad_average_corrected = grad_average ./ (1 - (1 - beta_momentum)^t);

        S = gamma * S + (1-gamma) * Hhat;
        S_corrected = S ./ (1 - (gamma)^t);
        w = w - ss*( (S_corrected + eye(D)./alpha) \ grad_average_corrected);

        w_all(:,t,m) = w(:);
        Sigma_all(:,:,t,m) = inv( S_corrected + eye(D)./alpha );
     otherwise
         error('no such method');
     end
   end
end

% save the experiment.
file_name = strcat('./toy_example_experiment_data.mat');
save(file_name, 'method', 'dataset_name', 'Sigma_all', 'w_all', 'w1', 'w2', 'W', 'f', 'Log_Prior', 'Log_Like', 'Log_Joint', 'post', 'wmap', 'w_exact_vi', 'C_exact_vi');

% plot and save figure 2 (a)
make_fig_two_a
