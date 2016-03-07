%% Test 1: Normal 25% data partitioning
clear all; clc;
load fisheriris;
X = meas;
Y = species;
instPart = PartitionDataSet(X,Y);

% Get 15% for test
instPart.doSimplePartitioning(15,1);
[tX,tY,vX,vY] = instPart.getDataset(1);

%% Test 2: K(5) partitioning
rng(1);
someDataset = [1:1:50]';
someDatasetX = [someDataset someDataset];
someDatasetY = someDataset;
kPart = cvpartition(someDatasetY,'k',5);
kPart.disp;
instPart = PartitionDataSet(someDatasetX,someDatasetY);
instPart.doKPartitioning(5,0);
[tX,tY,vX,vY] = instPart.getDataset(1);