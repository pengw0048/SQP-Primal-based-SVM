% see test_3x3 for full comments
% test solve_nlp_pchol on a 8*8 chessboard
n=1000;
V=rand(2,n)*8;    % [0,3]^2
a=ones(n,1);
for i=1:n
    if mod(floor(V(1,i))+floor(V(2,i)),2)==0
        a(i)=-1;
    end
end
% train svm
[w,b,h,U]=solve_nlp_pchol(a,V,10^5,10^-5,1,10^-5,20);
% draw the chessboard
figure
set(gcf,'position',[50,50,600,600])
hold on
for i=0:7
    for j=0:7
        if mod(i+j,2)==0
fill([0 1 1 0]+i,[0 0 1 1]+j,'y')
        end
    end
end
% draw zero contour - blue
x=linspace(0,8);
y=linspace(0,8);
[X,Y]=meshgrid(x,y);
Z=zeros(100,100);
for i=1:100
    for j=1:100
        [~,Z(i,j)]=classify_nlp(w,b,U,V,[X(i,j);Y(i,j)],1);
    end
end
contour(X,Y,Z,[0 0],'b','LineWidth',2)
% draw data points
miscl=0;
sv=0;
for i=1:n
    if a(i)==-1
        plot(V(1,i),V(2,i),'o')
    else
        plot(V(1,i),V(2,i),'x')
    end
    [a1,f]=classify_nlp(w,b,U,V,[V(1,i);V(2,i)],1);
    if a1~=a(i)
        plot(V(1,i),V(2,i),'+r')
        miscl=miscl+1;
    else
        if abs(abs(f)-abs(h))<10^-5
            sv=sv+1;
            plot(V(1,i),V(2,i),'sr')
        end
    end
end
axis equal
axis([0 8 0 8])
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