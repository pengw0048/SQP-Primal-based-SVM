% solve_sp
%   SQP-like algo in Sec.4 (Fletcher) with naive approach
% author: wp
% input: a,v,Delta,tau_d,maxiter
% output: w*,b*,h*
function [w,b,h]=solve_sp(a,V,Delta,taud,maxiter)
m=size(a,1);    % number of data points
n=size(V,1);    % dimension of v
AVt=diag(a)*V'; % pre-calculation
k=0;            % number of iters
d=taud*2;
% initialization
w=rand(n,1);    % "w_0 is arbitrary"
%w=ones(n,1);
w=w./norm(w);   % normalization
lpf=[0,-1];     % parameters in LP
lpA=[-a,ones(m,1)];
lpb=AVt*w;
options=optimoptions(@linprog,'display','none');    % suppress output
x=linprog(lpf,lpA,lpb,[],[],[],[],[],options);      % choose b_0 and h_0 by LP, x=[b_0;h_0]
b=x(1);
h=x(2);
% iteration
while norm(d)>=taud                 % criteria of termination
    pik=max([0 h]);                 % avoiding negative h_k (thus pi_k)
    qpH=diag([ones(n,1)*pik;0;0]);  % Hessian, parameters in QP_k
    qpf=[zeros(n+1,1);-1];
    qpA=[-AVt,-a,ones(m,1)];
    qpb=AVt*w+b.*a-h.*ones(m,1);
    qpAeq=[w',0,0];
    qpbeq=0;
    qplb=[-ones(n,1)*Delta;-inf;-inf];
    qpub=-qplb;
    options=optimoptions(@quadprog,'display','none');    % suppress output
    d=quadprog(qpH,qpf,qpA,qpb,qpAeq,qpbeq,qplb,qpub,[],options);  % solve QP_k, d=[dw;db;dh];
    dw=d(1:n);
    db=d(n+1);
    dh=d(n+2);
    wo=w+dw;
    bo=b+db;
    ho=h+dh;
    normw=norm(w);
    w=wo./normw;                    % w_k+1
    b=bo./normw;                    % b_k+1
    h=ho./normw;                    % h_k+1
    k=k+1;
    if k>maxiter
        fprintf('Too many iterations!\n')
        break
    end
    %norm(d)
end
k