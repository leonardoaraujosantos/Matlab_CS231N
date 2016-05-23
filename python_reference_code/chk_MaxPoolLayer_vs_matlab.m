%% Test Python version vs Matlab (Convlayer)
% On this script we will test the naive convolutional layer on our system
% against the cs231n version


%% Python preparation
% Add on python path assigment 2 (With teacher softmax)
% On the cs231n assigments 2 directory
clear all; clc;
insert(py.sys.path,int32(0),[pwd filesep 'python_reference_code' ...
    filesep 'cs231n_2016_solutions' filesep ...
    'assignment2' filesep 'cs231n']);
py.importlib.import_module('layers');

%% Load data (Forward propagation)
load maxPoolForward

%% Call python reference code (MaxPool Forward Propagation)
python_MAXPOOL_FP = cell(py.layers.max_pool_forward_naive(matArray2Numpy(x), pool_param));
python_MAXPOOL_FP_out = numpyArray2Mat(python_MAXPOOL_FP{1});
python_MAXPOOL_FP_cache = python_MAXPOOL_FP{2};
python_MAXPOOL_FP_cache_x = numpyArray2Mat(python_MAXPOOL_FP_cache{1});
python_MAXPOOL_FP_cache_params = python_MAXPOOL_FP_cache{2};

% Calculate errors
error_out = sum(abs(out(:) - python_MAXPOOL_FP_out(:)));
error_correct_out = sum(abs(correct_out(:) - python_MAXPOOL_FP_out(:)));

% Detect if error is bigger than 0.0000001
if (error_out > 1e-7) || (error_correct_out > 1e-7)
    fprintf('Python (MaxPool FP) calculation is wrong\n');
else
    fprintf('Python (MaxPool FP) calculation is right\n');
end


%% Call Matlab version (MaxPool Forward Propagation)
kernelSize = pool_param.pool_width;
stride = pool_param.stride;
maxPoolMat = MaxPoolingLayer(kernelSize, stride);

% Convert arrays from numpy to matlab multidimension format
% [rows,cols,dim1,... dimn] (Images comes transposed)
x_matlab = permute(x,[3 4 2 1]);
% The input are twp 4x4x3 image (4x4x3x2) after pooling it should be a 2
% images 2x2x3 (2x2x3x2) if you consider a 2x2 with stride 2 
mat_MAXPOOL_FP_OUT = maxPoolMat.feedForward(x_matlab);

% Permute python results to the same order as matlab
python_MAXPOOL_FP_out_perm = permute(python_MAXPOOL_FP_out,[3 4 2 1]);

% Permute matlab results to the same order as python
mat_MAXPOOL_FP_OUT_perm = permute(mat_MAXPOOL_FP_OUT,[4 3 1 2]);

% Calculate errors
error_out = sum(abs(out(:) - mat_MAXPOOL_FP_OUT_perm(:)));
error_correct_out = sum(abs(correct_out(:) - mat_MAXPOOL_FP_OUT_perm(:)));

% Detect if error is bigger than 0.0000001
if (error_out > 1e-7) || (error_correct_out > 1e-7)
    fprintf('Matlab (MaxPool FP) calculation is wrong\n');
else
    fprintf('Matlab (MaxPool FP) calculation is right\n');
end

%% Load data (Backward propagation)
clear all;
load maxPoolBackward

%% Call python reference code (MaxPool Forward Propagation)
pythonCacheTup = py.tuple({matArray2Numpy(cache{1}),cache{2}});
python_MAXPOOL_BP = py.layers.max_pool_backward_naive(matArray2Numpy(dout), pythonCacheTup);
python_MAXPOOL_BP = numpyArray2Mat(python_MAXPOOL_BP);

% Calculate errors
error_out = sum(abs(dx(:) - python_MAXPOOL_BP(:)));
error_correct_out = sum(abs(dx_num(:) - python_MAXPOOL_BP(:)));

% Detect if error is bigger than 0.0000001
if (error_out > 1e-7) || (error_correct_out > 1e-7)
    fprintf('Python (MaxPool BP) calculation is wrong\n');
else
    fprintf('Python (MaxPool BP) calculation is right\n');
end

%% Call Matlab version (MaxPool Backward Propagation)
kernelSize = pool_param.pool_width;
stride = pool_param.stride;
maxPoolMat = MaxPoolingLayer(kernelSize, stride);

% Convert arrays from numpy to matlab multidimension format
% [rows,cols,dim1,... dimn] (Images comes transposed)
x_matlab = permute(x,[3 4 2 1]);
dout_matlab = permute(dout,[3 4 2 1]);

% Load previous state
maxPoolMat.activations = x_matlab;

% Backpropagate
mat_MAXPOOL_BP_OUT = maxPoolMat.backPropagate(dout_matlab);

% Permute matlab results to the same order as python
mat_MAXPOOL_BP_OUT_perm = permute(mat_MAXPOOL_BP_OUT,[4 3 1 2]);

% Calculate errors
error_out = sum(abs(dx(:) - mat_MAXPOOL_BP_OUT_perm(:)));
error_correct_out = sum(abs(dx_num(:) - mat_MAXPOOL_BP_OUT_perm(:)));

% Detect if error is bigger than 0.0000001
if (error_out > 1e-7) || (error_correct_out > 1e-7)
    fprintf('Matlab (MaxPool BP) calculation is wrong\n');
else
    fprintf('Matlab (MaxPool BP) calculation is right\n');
end
