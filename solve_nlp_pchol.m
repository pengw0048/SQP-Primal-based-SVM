% solve_nlp_pchol
%   Train non-linear SVMs with partial Cholesky factorization (5.10)
% author: wp
% input: a,v,Delta,tau_d,sigma,tol,maxiter
% output: w*,b*,h*,U
function [w,b,h,U]=solve_nlp_pchol(a,V,Delta,taud,sigma,tol,maxiter)
K=gaussian_kernel(V,V,sigma);       % get K with gaussian kernel
n=size(K,1);
% the following routine is in the original paper, but erratic and unrecoverable presently
% % partial Cholesky factorization (5.10)
% d=diag(K);
% U=[];
% while 1
%     [~,i]=max(d);
%     if d(i)<tol
%         break
%     end
%     s=sqrt(d(i));
%     u=(K(i,:)-K(i,:)'*K)/s;
%     U=[U;u];
%     d=d-u.^2;
% end
[L,B,~,~] = partchol(K);	% get LDLt decomposition, B=D
for i=1:n
    if B(i,i)<tol			% we know the diagnoal elements of B are in descent order
        i=i-1;
        break;
    end
end
L1=sqrt(B)*L';				% get partial LLt decomposition
U=L1(1:i,:);
size(U,1)					% display the rank (size) of U
[w,b,h]=solve_sp(a,U,Delta,taud,maxiter);   % "then the SP is solved with U replacing V"
