% see test_3x3 for full comments
% test l1qd-svm on a 3*3 chessboard
n=100;
V=rand(2,n)*3;    % [0,3]^2
a=ones(n,1);
for i=1:n
    if mod(floor(V(1,i))+floor(V(2,i)),2)==0
        a(i)=-1;
    end
end
% train svm
SVMModel=fitcsvm(V',a,'KernelFunction','rbf','solver','L1QP','boxconstraint',1000);
% draw the chessboard
figure
set(gcf,'position',[200,200,300,300])
hold on
fill([0 1 1 0],[0 0 1 1],'y')
fill([0 1 1 0]+2,[0 0 1 1],'y')
fill([0 1 1 0],[0 0 1 1]+2,'y')
fill([0 1 1 0]+1,[0 0 1 1]+1,'y')
fill([0 1 1 0]+2,[0 0 1 1]+2,'y')
% draw zero contour - blue
x=linspace(0,3);
y=linspace(0,3);
[X,Y]=meshgrid(x,y);
Z=zeros(100,100);
for i=1:100
    for j=1:100
        [a1,b1]=predict(SVMModel,[X(i,j),Y(i,j)]);
            Z(i,j)=b1(1,2);
    end
end
contour(X,Y,Z,[0 0],'b','LineWidth',2)
% % draw +-h contour - cyan
% contour(X,Y,Z,[-h h],'c')
% draw data points
sv=0;
miscl=0;
for i=1:n
    if a(i)==-1
        plot(V(1,i),V(2,i),'o')
    else
        plot(V(1,i),V(2,i),'x')
    end
    [a1,b1]=predict(SVMModel,[V(1,i),V(2,i)]);
    if a1~=a(i)
        plot(V(1,i),V(2,i),'+r')
        miscl=miscl+1;
    else
        if SVMModel.IsSupportVector(i)
            plot(V(1,i),V(2,i),'sr')
            sv=sv+1;
        end
    end
end
axis equal
axis([0 3 0 3])
hold off
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