function C=unpackcovariance(L,D)

  C = zeros(D);
  C(triu(ones(D))==1)=L;
  C=C+triu(C,1)';
