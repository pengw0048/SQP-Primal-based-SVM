% test solve_nlp_chol with random nonlinear separable points
n=50;
V=rand(2,n)*2-1;    % [-1,1]^2
a=ones(n,1);
for i=1:n
    if norm(V(:,i))<0.6     % a circle centered at origin with radius 0.6
        a(i)=-1;
    end
end
% n=30;
% a=(rand(n,1)>=0.5)*2-1;
% V=rand(2,n).*[a a]';
% train svm
SVMModel=fitcsvm(V',a,'KernelFunction','rbf','solver','L1QP','boxconstraint',10000);
% draw zero contour - blue
x=linspace(-1,1);
y=linspace(-1,1);
[X,Y]=meshgrid(x,y);
Z=zeros(100,100);
for i=1:100
    for j=1:100
        [a1,b1]=predict(SVMModel,[X(i,j),Y(i,j)]);
            Z(i,j)=b1(1,2);
    end
end
figure
set(gcf,'position',[200,200,300,300])
contour(X,Y,Z,[0 0],'b')
hold on
% draw data points
for i=1:n
    if a(i)==-1
        plot(V(1,i),V(2,i),'o')
    else
        plot(V(1,i),V(2,i),'x')
    end
    [a1,b1]=predict(SVMModel,[V(1,i),V(2,i)]);
    if a1~=a(i)
        plot(V(1,i),V(2,i),'^r')
    else
        if SVMModel.IsSupportVector(i)
            plot(V(1,i),V(2,i),'sr')
        end
    end
end
% draw a circle (the real contour) - magenta
alpha=linspace(0,2*pi,100);
x=0.6*cos(alpha);
y=0.6*sin(alpha);
plot(x,y,'m');
axis equal
axis([-1 1 -1 1])
hold off