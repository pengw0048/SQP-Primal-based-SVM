% classify_nlp
%   Classify given point with trained nlp model
% author: wp
% input: w*,b*,U,V,v,sigma
% output: class (+-1),f
function [class,f]=classify_nlp(w,b,U,V,v,sigma)
rv=gaussian_kernel(V,v,sigma);          % rv=(K(v_1,v),...,K(v_m,v))'
theta=U'\rv;                            % the least-square solution of the system U' theta=rv
                                        % the original paper has a typo here
f=w'*theta+b;							% calculate f and classification
if f>=0
    class=1;
else
    class=-1;
end
