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
load ConvForward

%% Call python reference code (affine_forward)
% The functions matArray2Numpy and numpyArray2Mat do the job of converting
% a matlab array to a numpy array and bring a numpy array to matlab array
python_CONV_FP = cell(py.layers.conv_forward_naive(matArray2Numpy(x), matArray2Numpy(w), matArray2Numpy(b'), conv_param));
python_CONV_FP_OUT = numpyArray2Mat(python_CONV_FP{1});
% 
% % Just display
fprintf('External Python CS231n FullyConnected(forward) reference\n');
disp(python_CONV_FP_OUT);

%% Call matlab custom version (InnerProductLayer.forward_prop)
convMat = ConvLayer();
fpOutMat = fpMat.feedForward(x,w,b);
disp('Calculated on matlab');
disp(fpOutMat);
disp('CS231n reference (mat file)');
disp(correct_out);


%% Check if they are equal (InnerProductLayer Forward propagation)
error = abs(fpOutMat - out);
error = sum(error(:));
if error > 1e-8
    fprintf('Matlab (FullyConnected FP) calculation is wrong\n');
else
    fprintf('Matlab (FullyConnected FP) calculation is right\n');
end

%% Now test the back-propagation part
clear all;
load ConvBackward;

% %% Call python reference code (affine_backward)
% python_FC_BP = cell(py.layers.affine_backward(matArray2Numpy(dout), py.tuple({matArray2Numpy(cache{1}),matArray2Numpy(cache{2}),matArray2Numpy(cache{3})})));
% pyDx = numpyArray2Mat(python_FC_BP{1});
% pyDw = numpyArray2Mat(python_FC_BP{2});
% pyDb = numpyArray2Mat(python_FC_BP{3});
% 
% % Check if external python call went right
% if isequal(pyDx,dx) && isequal(pyDw,dw) && isequal(pyDb,db)
%     disp('External python call(affine_backward) is right');
% end
% 
% %% Call matlab custom version (InnerProductLayer.backPropagate)
% fpMat = InnerProductLayer();
% fpOutMatFP = fpMat.feedForward(x,w,b);
% [mat_dx,mat_dw,mat_db] = fpMat.backPropagate(dout);
% disp('Calculated on matlab dx');
% error_dx = abs(mat_dx - dx);
% error_dw = abs(mat_dw - dw);
% error_db = abs(mat_db - db);
% 
% error_dx = sum(error_dx(:));
% error_dw = sum(error_dw(:));
% error_db = sum(error_db(:));
% 
% if (error_dx > 1e-8) || (error_dw > 1e-8) || (error_db > 1e-8)
%     fprintf('Matlab (FullyConnected BP) calculation is wrong\n');
% else
%     fprintf('Matlab (FullyConnected BP) calculation is right\n');
% end
% 
% 
