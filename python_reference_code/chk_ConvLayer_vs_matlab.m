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

%% Define data
load ConvImages

% Get the image
image1 = reshape(x(1,:,:,:),[3 200 200]);
image2 = reshape(x(2,:,:,:),[3 200 200]);
% Transpose to put on matlab format, this is needed for color
image1 = permute(image1,[2,3,1]);
image2 = permute(image2,[2,3,1]);

figure(1);
hold on;
subplot(5,2,1);imshow(uint8(image1)); title('Dog');
subplot(5,2,2);imshow(uint8(image2)); title('Cat');

conv_param.stride = uint8(1);
conv_param.pad = uint8(1);

%% Call python reference code (convolution)
python_CONV_FP = cell(py.layers.conv_forward_naive(matArray2Numpy(x), matArray2Numpy(w), matArray2Numpy(b'), conv_param));
python_CONV_FP_OUT = numpyArray2Mat(python_CONV_FP{1});
image_out_1 = reshape(python_CONV_FP_OUT(1,1,:,:),[200 200]);
image_out_2 = reshape(python_CONV_FP_OUT(2,1,:,:),[200 200]);
image_out_3 = reshape(python_CONV_FP_OUT(1,2,:,:),[200 200]);
image_out_4 = reshape(python_CONV_FP_OUT(2,2,:,:),[200 200]);

subplot(5,2,3);imshow(uint8(image_out_1)); title('Dog Gray');
subplot(5,2,4);imshow(uint8(image_out_2)); title('Cat Gray');
subplot(5,2,5);imshow(uint8(image_out_3)); title('Dog Edge');
subplot(5,2,6);imshow(uint8(image_out_4)); title('Cat Edge');

%% Call Matlab version
kernelSize = 3;
numFilters = 2;
stride = 1;
pad = 1;
convMat = ConvolutionalLayer(kernelSize, numFilters, stride, pad);

% Convert arrays from numpy to matlab multidimension format
% [rows,cols,dim1,... dimn] (Images comes transposed)
x_matlab = permute(x,[3 4 2 1]);
w_matlab = permute(w,[3 4 2 1]);
matlabResult = convMat.feedForward(x_matlab,w_matlab,double(b));

image_out_mat_1 = matlabResult(:,:,1,1);
image_out_mat_2 = matlabResult(:,:,2,1);
image_out_mat_3 = matlabResult(:,:,1,2);
image_out_mat_4 = matlabResult(:,:,2,2);

subplot(5,2,7);imshow(uint8(image_out_mat_1)); title('Dog Gray(mat)');
subplot(5,2,8);imshow(uint8(image_out_mat_2)); title('Cat Gray(mat)');
subplot(5,2,9);imshow(uint8(image_out_mat_3)); title('Dog Edge(mat)');
subplot(5,2,10);imshow(uint8(image_out_mat_4)); title('Cat Edge(mat)');

%% Define data
clear all;clc;
load ConvForward

%% Call python reference code (affine_forward)
% The functions matArray2Numpy and numpyArray2Mat do the job of converting
% a matlab array to a numpy array and bring a numpy array to matlab array
python_CONV_FP = cell(py.layers.conv_forward_naive(matArray2Numpy(x), matArray2Numpy(w), matArray2Numpy(b'), conv_param));
python_CONV_FP_OUT = numpyArray2Mat(python_CONV_FP{1});
%
% % Just display
fprintf('External Python CS231n FullyConnected(forward) reference\n');
error = sum(abs(out(:) - python_CONV_FP_OUT(:)));
fprintf('Difference on python (exteral) version is %d\n',error);

%% Call matlab custom version (InnerProductLayer.forward_prop)
kernelSize = 3;
numFilters = 4;
stride = 2;
pad = 1;
convMat = ConvolutionalLayer(kernelSize, numFilters, stride, pad);
% Convert arrays from numpy to matlab multidimension format
% [rows,cols,dim1,... dimn] (Images comes transposed)
x_matlab = permute(x,[3 4 2 1]);
w_matlab = permute(w,[3 4 2 1]);
matlabResult = convMat.feedForward(x_matlab,w_matlab,b);
% Put on same order has python
matlabResult = permute(matlabResult,[3 4 1 2]);
error = sum(abs(out(:) - matlabResult(:)));
fprintf('Difference with matlab version is %d\n',error);

%% Check if they are equal (InnerProductLayer Forward propagation)
if error > 1e-8
    fprintf('Matlab (Conv FP) calculation is wrong\n');
else
    fprintf('Matlab (Conv FP) calculation is right\n');
end


%% Now test the back-propagation part
clear all;
load ConvBackward;

%% Call python reference code (affine_backward)
pythonCacheTup = py.tuple({matArray2Numpy(cache{1}),matArray2Numpy(cache{2}),matArray2Numpy(cache{3}'),cache{4}});
python_FC_BP = cell(py.layers.conv_backward_naive(matArray2Numpy(dout), pythonCacheTup));
pyDx = numpyArray2Mat(python_FC_BP{1});
pyDw = numpyArray2Mat(python_FC_BP{2});
pyDb = numpyArray2Mat(python_FC_BP{3});

errorDx = sum(abs(pyDx(:)-dx(:)));
errorDw = sum(abs(pyDw(:)-dw(:)));
errorDb = sum(abs(pyDb(:)-db(:)));

if (errorDx > 1e-8) || (errorDb > 1e-8) || (errorDw > 1e-8)
    fprintf('Problem calling python conv backprop\n');
else
    fprintf('Python conv backprop is correct\n');
end

% Put 4d tensors on matlab format
dout_matlab = permute(dout,[3 4 2 1]);
prev_x = permute(cache{1},[3 4 2 1]);
prev_w = permute(cache{2},[3 4 2 1]);
prev_b = cache{3};

kernelSize = 3;
numFilters = 2;
stride = 1;
pad = 1;
convMat = ConvolutionalLayer(kernelSize, numFilters, stride, pad);

convMat.previousInput = prev_x;
convMat.weights = prev_w;
convMat.biasWeights = prev_b;

[matDx, matDw, matDb] = convMat.backPropagate(dout_matlab);

errorDx = sum(abs(matDx(:)-dx(:)));
errorDw = sum(abs(matDw(:)-dw(:)));
errorDb = sum(abs(matDb(:)-db(:)));

if (errorDx > 1e-8) || (errorDb > 1e-8) || (errorDw > 1e-8)
    fprintf('Matlab conv backprop calculation incorrect\n');
else
    fprintf('Matlab conv backprop calculation correct\n');
end
