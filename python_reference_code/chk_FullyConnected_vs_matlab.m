%% Test Python version vs Matlab (FullyConnect)
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
load dataTestFullyConnected_FP

%% Call python reference code (relu_forward)
% The functions matArray2Numpy and numpyArray2Mat do the job of converting
% a matlab array to a numpy array and bring a numpy array to matlab array
% pythonReluFPOut = cell(py.layers.relu_forward(matArray2Numpy(x)));
% pythonReluOut = numpyArray2Mat(pythonReluFPOut{1});
% pythonReluCache = numpyArray2Mat(pythonReluFPOut{2});
% 
% % Just display
% fprintf('Python CS231n FullyConnected(forward) reference\n');
% disp(pythonReluOut);
% disp(correct_fp);
% disp('Cache the input itself');
% disp(pythonReluCache);

%% Call matlab custom version (ReluActivation.forward_prop)
fpMat = InnerProductLayer();
fpOutMat = fpMat.feedForward(x,w,b);
disp(fpOutMat);


%% Check if they are equal (Relu Forward propagation)
error = abs(fpOutMat - out);
error = sum(error(:));
if error > 1e-7
    fprintf('Matlab (FullyConnected FP) calculation is wrong\n');
else
    fprintf('Matlab (FullyConnected FP) calculation is right\n');
end

