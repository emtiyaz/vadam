function L=packcovariance(C)

 inds = find(triu(ones(size(C))));
 L = C(inds);
 L=L(:);