%% Test 1: Normal 25% data partitioning
clear all; clc;
load fisheriris;
X = meas;
Y = species;
instPart = PartitionDataSet(X,Y);

% Get 15% for test
instPart.doSimplePartitioning(15,1);
[tX,tY,vX,vY] = instPart.getDataset(1);

%% Test 2: Check python support
pyversion
someDataset = [1:1:30]';
someDataSetX_Python = py.list(someDataset');
result = py.k_fold.test_k_fold(someDataSetX_Python,int32(5));
% Convert to cell then to matlab arrays
result = cell(result);
training = cell(result{1});
validation = cell(result{2});
training_from_python = cellfun(@double,training)';
validation_from_python = cellfun(@double,validation)';

%% Test 3: Compare sizes
someDataset = [1:1:30]';
someDatasetX = [someDataset someDataset];
someDatasetY = someDataset;
someDataSetY_Python = py.list(someDatasetY');
kPart = cvpartition(someDatasetY,'k',5);
kPart.disp;
instPart = PartitionDataSet(someDatasetX,someDatasetY);
instPart.doKPartitioning(5,0);
assert (kPart.NumObservations == instPart.getNumObservations);
assert (kPart.NumTestSets == instPart.getNumTestSets);
assert (kPart.TrainSize(1) == instPart.getTrainSize);
assert (kPart.TestSize(1) == instPart.getTestSize);

%% Test 4: K(5) partitioning
someDataset = [1:1:30]';
someDatasetX = [someDataset someDataset];
someDatasetY = someDataset;
someDataSetY_Python = py.list(someDatasetY');
kPart = cvpartition(someDatasetY,'k',5);
kPart.disp;
instPart = PartitionDataSet(someDatasetX,someDatasetY);
instPart.doKPartitioning(5,0);
[tX,tY,vX,vY] = instPart.getDataset(1);

% Get the trainning set Y from matlab fold 1
mat_ty = someDatasetY(kPart.training(1));
mat_vy = someDatasetY(kPart.test(1));

% Compare the vector content
same_size = (length(mat_vy) == length(vY));
content_equal = isequal(vY,mat_vy);
assert (same_size == 1);
assert (content_equal == 1);