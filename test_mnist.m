% see test_3x3 for full comments
% test solve_nlp_pchol with the mnist database
% just classify 1 (+1) and 9 (-1)
% pick the first n*20 digits from each image as the training data
% the next m*20 digits are used for testing
% each vector has 784 dimensions
n=20;
m=8;
sigma=50000;
im1=imread('mnist_test1.jpg');
im9=imread('mnist_test9.jpg');
V=zeros(784,n*40);
a=[ones(n*20,1);-ones(n*20,1)];
for i=1:n
    for j=1:20
        V(:,i*20-20+j)=reshape(double(im1(i*28-27:i*28,j*28-27:j*28)),784,1);
    end
end
for i=1:n
    for j=1:20
        V(:,(n+i)*20-20+j)=reshape(double(im9(n*28-27:n*28,j*28-27:j*28)),784,1);
    end
end
% train svm
[w,b,h,U]=solve_nlp_pchol(a,V,10^5,10^-5,sigma,10^-5,50);
% classify the training data
miscl=0;
sv=0;
for i=1:n*40
    [a1,f]=classify_nlp(w,b,U,V,V(:,i),sigma);
    if a1~=a(i)
        miscl=miscl+1;
    else
        if abs(abs(f)-abs(h))<10^-4
            sv=sv+1;
        end
    end
end
miscl
sv
% classify the testing data
T=zeros(784,m*40);
for i=1:m
    for j=1:20
        T(:,i*20-20+j)=reshape(double(im1((n+i)*28-27:(n+i)*28,j*28-27:j*28)),784,1);
    end
end
for i=1:m
    for j=1:20
        T(:,(m+i)*20-20+j)=reshape(double(im9((n+i)*28-27:(n+i)*28,j*28-27:j*28)),784,1);
    end
end
miscl=0;
for i=1:m*40
    [a1,f]=classify_nlp(w,b,U,V,T(:,i),sigma);
    if (a1==1&&i>m*20)||(a1==-1&&i<=m*20)
        miscl=miscl+1;
    end
end
miscl
