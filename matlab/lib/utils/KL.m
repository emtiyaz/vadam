function [dKL] = KL(p,q)
% Kullback Leibler divergence (KL) KL(p,q)
% Input:
%     p: p.mu, p.sigma
%     q: q.mu, q.sigma
% Output:
%     delta: the KL divergence between p and q.
% Signature
%   Author: Meizhu Liu
%   E-Mail: meizhu.liu@yahoo.com
%   Date  : March 30, 2014
% References:
% Meizhu Liu, Baba C. Vemuri, Shun-Ichi Amari and Frank Nielsen,
% ?Shape Retrieval Using Hierarchical Total Bregman Soft Clustering?,
% Transactions on Pattern Analysis and Machine Intelligence (TPAMI), 2012
%
% Baba C. Vemuri, Meizhu Liu, Shun-Ichi Amari and Frank Nielsen,
%   Total Bregman Divergence and its Applications to DTI Analysis,
%   IEEE Transactions on Medical Imaging (TMI'10), 2010.
%
%   Meizhu Liu, Baba C. Vemuri, Shun-Ichi Amari and Frank Nielsen,
%   Total Bregman Divergence and its Applications to Shape Retrieval,
%   IEEE Conference on Computer Vision and Pattern Recognition (CVPR'10),
%   2010.
% Example Usage:
%     p.mu = [1 2]';
%     p.sigma = [1 0; 0 9];
%     q.mu = [0 1]';
%     q.sigma = [4 0; 0 16];
%     dKL = KL(p,q);


mu0 = p.mu;
mu1 = q.mu;
k = length(mu0);
sigma0 = p.sigma;
sigma1 = q.sigma;


tmp = inv(sigma1)*sigma0;
dKL = 0.5*(trace(tmp)+(mu1-mu0)'*inv(sigma1)*(mu1-mu0)-k-log(det(tmp)));
