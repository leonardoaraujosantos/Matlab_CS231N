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

%% Load data
load maxPoolForward

%% Call python reference code (MaxPool)
python_MAXPOOL_FP = cell(py.layers.max_pool_forward_naive(matArray2Numpy(x), pool_param));
python_MAXPOOL_FP_out = numpyArray2Mat(python_MAXPOOL_FP{1});
python_MAXPOOL_FP_cache = python_MAXPOOL_FP{2};
python_MAXPOOL_FP_cache_x = numpyArray2Mat(python_MAXPOOL_FP_cache{1});
python_MAXPOOL_FP_cache_params = python_MAXPOOL_FP_cache{2};

% Calculate errors
error_out = sum(abs(out(:) - python_MAXPOOL_FP_out(:)));
error_correct_out = sum(abs(correct_out(:) - python_MAXPOOL_FP_out(:)));

if (error_out > 1e-8) && (error_correct_out > 1e-8)
    fprintf('Python (MaxPool FP) calculation is wrong\n');
else
    fprintf('Python (MaxPool FP) calculation is right\n');
end


%% Call Matlab version
kernelSize = pool_param.pool_width;
stride = pool_param.stride;
maxPoolMat = MaxPoolingLayer(kernelSize, stride);

% Convert arrays from numpy to matlab multidimension format
% [rows,cols,dim1,... dimn] (Images comes transposed)
x_matlab = permute(x,[3 4 2 1]);
mat_MAXPOOL_FP_OUT = maxPoolMat.feedForward(x_matlab);
mat_MAXPOOL_FP_OUT = permute(mat_MAXPOOL_FP_OUT,[3 4 2 1]);

% Calculate errors
error_out = sum(abs(out(:) - mat_MAXPOOL_FP_OUT(:)));
error_correct_out = sum(abs(correct_out(:) - mat_MAXPOOL_FP_OUT(:)));

if (error_out > 1e-8) && (error_correct_out > 1e-8)
    fprintf('Matlab (MaxPool FP) calculation is wrong\n');
else
    fprintf('Matlab (MaxPool FP) calculation is right\n');
end



