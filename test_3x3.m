% test solve_nlp_(p)chol on a 3*3 chessboard
% generate random points
sigma=1;
n=100;
V=rand(2,n)*3;    % [0,3]^2
a=ones(n,1);
for i=1:n
    if mod(floor(V(1,i))+floor(V(2,i)),2)==0
        a(i)=-1;
    end
end
% pick some random points at the border and mislabel them
% uncomment the following lines if you want this feature
dist=0.1;				% the maximal distance to border for choosing mislabelled points
misl=zeros(n,1);		% whether it is mislabelled
% for i=1:n
%     if (ceil(V(1,i))-V(1,i)<dist || V(1,i)-floor(V(1,i))<dist || ceil(V(2,i))-V(2,i)<dist || V(2,i)-floor(V(2,i))<dist ) && V(1,i)>0.2 && V(1,i)<2.8 && V(2,i)>0.2 && V(2,i)<2.8 && rand()>0.4
%         a(i)=-a(i);
%         misl(i)=1;
%     end
% end
% sum(misl)
% train svm
[w,b,h,U]=solve_nlp_pchol(a,V,10^5,10^-5,sigma,10^-5,100);
% draw the chessboard
figure
set(gcf,'position',[200,200,300,300])	% just for exporting figures with fixed size
hold on
fill([0 1 1 0],[0 0 1 1],'y')
fill([0 1 1 0]+2,[0 0 1 1],'y')
fill([0 1 1 0],[0 0 1 1]+2,'y')
fill([0 1 1 0]+1,[0 0 1 1]+1,'y')
fill([0 1 1 0]+2,[0 0 1 1]+2,'y')
% draw zero contour - blue
% test f values at mesh-grid points
x=linspace(0,3);
y=linspace(0,3);
[X,Y]=meshgrid(x,y);
Z=zeros(100,100);
for i=1:100
    for j=1:100
        [~,Z(i,j)]=classify_nlp(w,b,U,V,[X(i,j);Y(i,j)],sigma);
    end
end
contour(X,Y,Z,[0 0],'b','LineWidth',2)
% draw +-h contour - cyan
contour(X,Y,Z,[-h h],'c')
% draw data points
sv=0;
miscl=0;
for i=1:n
    if a(i)==-1
        plot(V(1,i),V(2,i),'o')
    else
        plot(V(1,i),V(2,i),'x')
    end
    [a1,f]=classify_nlp(w,b,U,V,[V(1,i);V(2,i)],sigma);		% decide if this point is misclassified
    if a1~=a(i)
        plot(V(1,i),V(2,i),'+r')
        miscl=miscl+1;
    else
        if abs(abs(f)-abs(h))<10^-4							% this criterion is not reliable. should find ways to identify SVs properly
            sv=sv+1;
            plot(V(1,i),V(2,i),'sr')
        end
    end
    if misl(i)==1
        plot(V(1,i),V(2,i),'rd','MarkerSize',12)
    end
end
axis equal
axis([0 3 0 3])
hold off
% measure test error
wcount=0;
for i=1:100
    for j=1:100
        if mod(floor(X(i,j))+floor(Y(i,j)),2)==0
            label=-1;
        else
            label=1;
        end
        if label~=sign(Z(i,j))
            wcount=wcount+1;
        end
    end
end
sv
miscl
wcount