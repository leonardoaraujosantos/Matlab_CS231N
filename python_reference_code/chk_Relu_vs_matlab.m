%% Test Python version vs Matlab (Relu)
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
x = [-0.5,-0.40909091,-0.31818182,-0.22727273; -0.13636364,-0.04545455,0.04545455,0.13636364; 0.22727273,0.31818182,0.40909091,0.5];
dout = [0.6503242 , -0.86276676, -0.41114615,  1.15480559; -0.04510508,  0.42359947,  1.13203137, -0.30135041; -1.45744589,  0.53074212,  1.28874525,  0.32282054];
correct_fp = [0 0 0 0; 0 0 0.04545455 0.13636364; 0.22727273 0.31818182,  0.40909091,  0.5 ];

%% Call python reference code (relu_forward)
% The functions matArray2Numpy and numpyArray2Mat do the job of converting
% a matlab array to a numpy array and bring a numpy array to matlab array
pythonReluFPOut = cell(py.layers.relu_forward(matArray2Numpy(x)));
pythonReluOut = numpyArray2Mat(pythonReluFPOut{1});
pythonReluCache = numpyArray2Mat(pythonReluFPOut{2});

% Just display
fprintf('Python CS231n Relu(forward) reference\n');
disp(pythonReluOut);
disp(correct_fp);
disp('Cache the input itself');
disp(pythonReluCache);

%% Call matlab custom version (ReluActivation.forward_prop)
relu = ReluLayer();
matReluOut = relu.feedForward(x);
disp(matReluOut);


%% Check if they are equal (Relu Forward propagation)
if ~isequal(matReluOut,pythonReluOut)
    fprintf('Matlab (Relu-forward) calculation is wrong\n');
else
    fprintf('Matlab (Relu-forward) calculation is right\n');
end


%% Call python reference code (relu_backward)
% The functions matArray2Numpy and numpyArray2Mat do the job of converting
% a matlab array to a numpy array and bring a numpy array to matlab array
pythonReluBPOut = py.layers.relu_backward(matArray2Numpy(dout), matArray2Numpy(pythonReluCache));
pythonReluBPOut = numpyArray2Mat(pythonReluBPOut);

% Just display
fprintf('Python CS231n Relu(backward) reference\n');
disp(pythonReluBPOut);


%% Call matlab custom version (ReluActivation.back_prop)
matReluOut = relu.backPropagate(dout);
disp(matReluOut);

%% Check if they are equal (Relu Backward propagation)
if ~isequal(matReluOut,pythonReluBPOut)
    fprintf('Matlab (Relu-forward) calculation is wrong\n');
else
    fprintf('Matlab (Relu-forward) calculation is right\n');
end