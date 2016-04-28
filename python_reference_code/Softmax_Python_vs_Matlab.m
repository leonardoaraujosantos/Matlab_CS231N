%% Test Python version vs Matlab
% On this script we will test the softmax code reference from cs231n
% assigment with our own softmax version, both should return the loss and
% the gradients related to the target input, both versions receive the same
% data.
% 
% <<../../docs/imgs/SVM_vs_Softmax.png>>
%

%% Reference code on Python
% Basically this piece of code does the same thing but on python
% Code:
%
% <include>TestSoftMax.py</include>
%

%% Python preparation
% Add on python path assigment 2 (With teacher softmax)
% On the cs231n assigments 2 directory
clear all; clc;
insert(py.sys.path,int32(0),[pwd filesep 'python_reference_code' ...
    filesep 'cs231n_2016_solutions' filesep ...
    'assignment2' filesep 'cs231n']);
py.importlib.import_module('layers');

%% Define data
scores = [-2.85, 0.86, 0.28; 0 0.3 0.2; 0.3 -0.3 0];
correct = [3 2 1];
correctOneHot = oneHot(correct');

%% Call python reference code
% The functions matArray2Numpy and numpyArray2Mat do the job of converting
% a matlab array to a numpy array and bring a numpy array to matlab array
pythonSoftmax = cell(py.layers.softmax_loss(...
    matArray2Numpy(scores),matArray2Numpy(uint8(correct-1))));
pythonLoss = pythonSoftmax{1};
pythonDw = numpyArray2Mat(pythonSoftmax{2});

% Just display
fprintf('Python CS231n softmax reference\n');
disp(pythonLoss);
disp(pythonDw);

%% Call matlab custom version
testLossFunction = SoftMaxLoss();
[matLoss, matDw] = testLossFunction.getLoss(scores,correctOneHot);
fprintf('Matlab version\n');
disp(matLoss);
disp(matDw);

%% Check if they are equal
if ~isequal(matLoss,pythonLoss)
    fprintf('Matlab loss calculation is wrong\n');
else
    fprintf('Matlab loss calculation is right\n');
end

if ~isequal(pythonDw,matDw)
    fprintf('Matlab derivative calculation is wrong\n');
else
    fprintf('Matlab derivative calculation is right\n');
end