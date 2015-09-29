% gaussian_kernel
%   Gaussian kernel for two matricies of column vectors
% author: wp
% input: matricies U,V,sigma
% output: Gram matrix G
function G=gaussian_kernel(U,V,sigma)
% inspired by http://stackoverflow.com/questions/13109826/compute-a-gramm-matrix-in-matlab-without-loops
sizeU=size(U,2);
sizeV=size(V,2);
% the formula below is derived from expanding ||u-v|| to avoid loops
A=repmat(sum(U.*U,1)',1,sizeV);
B=-2*U'*V;
C=repmat(sum(V.*V,1),sizeU,1);
G=A+B+C;
G=exp(-G/(2*sigma^2));
