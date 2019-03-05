clear
clc

dataname='/home/changjianlong/datasets/STL10.h5';
y = h5read(dataname, '/y_train')+1; % labels in dataset

filename='/home/changjianlong/clustering/improve/STL-10/stl_8.h5';
acc = h5read(filename, '/acc'); % clustering accuracies in all epoches
out = h5read(filename, '/output'); % results in all epoches
tempmap = h5read(filename, '/tempmap');

out1 = out;

for i=1:size(tempmap,1)
    out(tempmap(i)+1)=out1(i);
end


c = [1,0,0;
    0,1,0;
    0,0,1;
    1,1,0;
    1,0,1;
    0,1,1;
    0,0,0;
    0,0.5,0.5;
    0.5,0,0.5;
    0.5,0.5,0];
k = eps;

ind = 1; % original state

one = reshape(out(:,:,ind),size(out,1),size(out,2));
one = one';
one = one./repmat(sum(one,2),1,size(out,1));

nb_classes = size(one,2);
theta = (0:36:359)/180*pi;
points = [sin(theta)',cos(theta)']; % high dim vectors to 2D points
onet = one*points;

nb = 13000; % number of points


figure,
for i = 1:nb
    i
    plot(onet(i:i,1),onet(i:i,2),'.','LineWidth',1,'MarkerSize',10,'Color',c(y(tempmap(i)+1),:)),hold on
end
axis([-1,1,-1,1])
set(gca,'position',[0,0,1,1],'box','off','xtick',[],'ytick',[],'xcolor','w','ycolor','w')