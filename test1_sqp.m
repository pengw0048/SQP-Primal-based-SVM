% test solve_nlp_(p)chol with random linear separable points
n=30;
V=rand(2,n)*2-1;
a=ones(n,1);
for i=1:n
    if V(1,i)+V(2,i)<0
        a(i)=-1;
    end
end
% train svm
[w,b,h,U]=solve_nlp_chol(a,V,10^5,10^-5,1,100);
% draw contour (f=0) - blue
x=linspace(-1,1);
y=linspace(-1,1);
[X,Y]=meshgrid(x,y);
Z=zeros(100,100);
for i=1:100
    for j=1:100
        [~,Z(i,j)]=classify_nlp(w,b,U,V,[X(i,j);Y(i,j)],1);
    end
end
figure
set(gcf,'position',[200,200,300,300])
contour(X,Y,Z,[0 0],'b');
hold on
% draw +-h contour - cyan
contour(X,Y,Z,[-h h],'c')
% draw points
for i=1:n
    if a(i)==-1
        plot(V(1,i),V(2,i),'o')
    else
        plot(V(1,i),V(2,i),'x')
    end
    [~,f]=classify_nlp(w,b,U,V,[V(1,i);V(2,i)],1);
    if abs(abs(f)-abs(h))<10^-3
        plot(V(1,i),V(2,i),'sr')
    end
end
axis equal
axis([-1 1 -1 1])
hold off
