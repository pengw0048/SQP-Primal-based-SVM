% solve_nlp_chol
%   Train non-linear SVMs with full Cholesky factorization
% author: wp
% input: a,v,Delta,tau_d,sigma,maxiter
% output: w*,b*,h*,U,IsSupporingVector
function [w,b,h,U,issv]=solve_nlp_chol(a,V,Delta,taud,sigma,maxiter)
K=gaussian_kernel(V,V,sigma);       % get K with gaussian kernel
U=chol(K);                          % Cholesky factorization K=U'U
[w,b,h]=solve_sp(a,U,Delta,taud,maxiter);   % "then the SP is solved with U replacing V"
% issv is currently not implemented
