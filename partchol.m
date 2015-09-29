% This function is by A. FORSGREN, P. E. GILL, AND W. MURRAY
%   in paper COMPUTING MODIFIED NEWTON DIRECTIONS USING A PARTIAL CHOLESKY FACTORIZATION

% PARTCHOL Partial Cholesky factorization routine for a real symmetric matrix H.
% [L,B,perm,n1] = partchol(H)
% forms a permutation perm, a unit lower-triangular matrix
% L(perm,:) and a block diagonal matrix B such that
% LBL'=H
% using the partial Cholesky factorization with diagonal pivoting.
% The size of the positive-definite principal submatrix obtained
% in the factorization is n1.
function [L,B,perm,n1] = partchol(H)
n = length(H);
perm = 1:n;
B = H;
L = zeros(n);
nu=0.5;
k = 1;
n1 = 0;
while k <= n
  [mur,r] = max([zeros(1,k-1) diag(B(k:n,k:n))']);
  if k < n
    mupr = max(abs(B(r,[1:r-1 r+1:n])));
  else
    mupr = 0;
  end
  if mur > 0 && mur >= nu * mupr
    n1 = k;
    perm([k r]) = perm([r k]);
    B([k r],:) = B([r k],:);
    B(:,[k r]) = B(:,[r k]);
    L(perm(k:n),k) = B(k:n,k)/B(k,k);
    if k < n
      B(k+1:n,k+1:n) = B(k+1:n,k+1:n)-L(perm(k+1:n),k)*B(k,k+1:n);
      B(k+1:n,k) = zeros(n-k,1);
      B(k,k+1:n) = zeros(1,n-k);
    end
    k = k+1;
  else
    L(perm(k:n),k:n) = eye(n-k+1);
    k = n+1;
  end
end
