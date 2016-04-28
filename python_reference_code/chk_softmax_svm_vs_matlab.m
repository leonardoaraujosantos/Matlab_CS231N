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

%% Call python reference code (softmax_loss)
% The functions matArray2Numpy and numpyArray2Mat do the job of converting
% a matlab array to a numpy array and bring a numpy array to matlab array
pythonSoftmax = cell(py.layers.softmax_loss(...
    matArray2Numpy(scores),(uint8(correct-1))));
pythonLoss = pythonSoftmax{1};
pythonDw = numpyArray2Mat(pythonSoftmax{2});

% Just display
fprintf('Python CS231n softmax reference\n');
disp(pythonLoss);
disp(pythonDw);

%% Call matlab custom version (SoftMaxLoss)
testLossFunction = SoftMaxLoss();
[matLoss, matDw] = testLossFunction.getLoss(scores,correctOneHot);
fprintf('Matlab Softmax version\n');
disp(matLoss);
disp(matDw);

%% Check if they are equal (Softmax)
if ~isequal(matLoss,pythonLoss)
    fprintf('Matlab (Softmax) loss calculation is wrong\n');
else
    fprintf('Matlab (Softmax) loss calculation is right\n');
end

if ~isequal(pythonDw,matDw)
    fprintf('Matlab (Softmax) derivative calculation is wrong\n');
else
    fprintf('Matlab (Softmax) derivative calculation is right\n');
end

%% Call python reference code (svm_loss)
% The functions matArray2Numpy and numpyArray2Mat do the job of converting
% a matlab array to a numpy array and bring a numpy array to matlab array
pythonSVMLoss = cell(py.layers.svm_loss(...
    matArray2Numpy((scores)),int32(correct-1)));
pythonLoss = pythonSVMLoss{1};
pythonDw = numpyArray2Mat(pythonSVMLoss{2});

% Just display
fprintf('Python CS231n svm_loss reference\n');
disp(pythonLoss);
disp(pythonDw);
 
%% Call matlab custom version (SVMLoss)
testLossFunction = SVMLoss(1);
[matLoss, matDw] = testLossFunction.getLoss(scores,correctOneHot);
fprintf('Matlab Softmax version\n');
disp(matLoss);
disp(matDw);

%% Check if they are equal (SVM loss)
if ~isequal(matLoss,pythonLoss)
    fprintf('Matlab (SVM) loss calculation is wrong\n');
else
    fprintf('Matlab (SVM) loss calculation is right\n');
end

if ~isequal(pythonDw,matDw)
    fprintf('Matlab (SVM) derivative calculation is wrong\n');
else
    fprintf('Matlab (SVM) derivative calculation is right\n');
end