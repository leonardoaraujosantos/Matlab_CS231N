%% Test Python version vs Matlab (FullyConnect)
% On this script we will test the fully connected layer on our system
% against the cs231n affine layer (affine==fully connected)


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

%% Create the same data on python exercise on matlab by hand
num_inputs=2;
% Notice the difference that on python (4,5,6) create a 4 dimension matrix
% with 5x6, the same thing on matlab will be (5,6,4)
input_shape = [5 6 4];
output_dim = 3;
input_size = num_inputs * prod(input_shape);
weight_size = output_dim * prod(input_shape);

x_py = py.numpy.linspace(-0.1,0.5,input_size);
x_py_conv = numpyArray2Mat(x_py);
x_mat = linspace(-0.1,0.5,input_size);
diffX = sum(abs(x_py_conv - x_mat));
fprintf('isequal(x_mat,x_py_conv)==%d\n',~(diffX > 1e-9));


% Also notice that the reshape on python is by default row-major, and on
% matlab is col-major
x_py = x_py.reshape(2,4,5,6);
x_py_conv = numpyArray2Mat(x_py);
fprintf('isequal(x,x_py_conv)==%d\n',isequal(x,x_py_conv));

w_py = py.numpy.linspace(-0.2,0.3,weight_size);
w_py_conv = numpyArray2Mat(w_py);
w_mat = linspace(-0.2,0.3,weight_size);
diffW = sum(abs(w_mat - w_py_conv));
fprintf('isequal(w_mat,w_py_conv)==%d\n',~(diffW > 1e-9));

% Don't forget if you want numerical match between python and matlab pay
% attention that the reshape operation on python is row-major and in matlab
% col-major. So instead of the following line:
%w_mat = reshape(w_mat,[prod(input_shape) output_dim]);
% do this....
% w_mat = reshape(w_mat,[output_dim prod(input_shape)])';
% Also don't forget that if the matrices involved has more than 2
% dimensions you need to "permute"
w_mat = reshape(w_mat,[output_dim prod(input_shape)])';
diffW = sum(abs(w_mat(:) - w(:)));
fprintf('isequal(w_mat(:),w(:))==%d\n',~(diffW > 1e-9));

% On ths multidimensional case, we create a matrix 2x4x5x6 on python, to
% have the same matrix on matlab we need to reshape to 6x5x4x2, then
% transpose(permute) all of it's dimensions
%x_mat = reshape(x_mat,[6 5 4 2]);
%x_mat = permute(x_mat,[ndims(x_mat):-1:1]);
% We create a helper function "reshape_row_major" that do the reshape on
% the same numpy order
x_mat = reshape_row_major(x_mat,[2,4,5,6]);
diffX = sum(abs(x_mat(:) - x(:)));
fprintf('isequal(x_mat(:),x(:))==%d\n',~(diffX > 1e-9));
% Now put on the matlab order
x_mat = permute(x_mat,[3,4,2,1]);

b_mat = linspace(-0.3,0.1,output_dim);
diffB = sum(abs(b_mat(:) - b(:)));

if (diffW < 1e-9) && (diffX < 1e-9) && (diffB < 1e-9)
    fprintf('Manual creation of the data on matlab passed\n');
else
    fprintf('Manual creation of the data on matlab failed\n');
end

%% Call python reference code (affine_forward)
% The functions matArray2Numpy and numpyArray2Mat do the job of converting
% a matlab array to a numpy array and bring a numpy array to matlab array
python_FC_FP = cell(py.layers.affine_forward(matArray2Numpy(x), matArray2Numpy(w), matArray2Numpy(b')));
pythonFC = numpyArray2Mat(python_FC_FP{1});
% 
% % Just display
fprintf('External Python CS231n FullyConnected(forward) reference\n');
disp(pythonFC);

%% Call matlab custom version (InnerProductLayer.forward_prop)
x_perm_mat = permute(x,[4,3,2,1]);
fpMat = InnerProductLayer(1);
fpOutMat = fpMat.feedForward(x_perm_mat,w,b);
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
load dataTestFullyConnected_BP;

%% Call python reference code (affine_backward)
python_FC_BP = cell(py.layers.affine_backward(matArray2Numpy(dout), py.tuple({matArray2Numpy(cache{1}),matArray2Numpy(cache{2}),matArray2Numpy(cache{3})})));
pyDx = numpyArray2Mat(python_FC_BP{1});
pyDw = numpyArray2Mat(python_FC_BP{2});
pyDb = numpyArray2Mat(python_FC_BP{3});

% Check if external python call went right
if isequal(pyDx,dx) && isequal(pyDw,dw) && isequal(pyDb,db)
    disp('External python call(affine_backward) is right');
end

%% Call matlab custom version (InnerProductLayer.backPropagate)
fpMat = InnerProductLayer(1);
x_perm_mat = permute(x,[3,2,1]);
fpOutMatFP = fpMat.feedForward(x_perm_mat,w,b);
[mat_dx,mat_dw,mat_db] = fpMat.backPropagate(dout);
%mat_dx = permute(mat_dx,[ndims(mat_dx):-1:1]);
disp('Calculated on matlab dx');
error_dx = abs(mat_dx - dx);
error_dw = abs(mat_dw - dw);
error_db = abs(mat_db - db);

error_dx = sum(error_dx(:));
error_dw = sum(error_dw(:));
error_db = sum(error_db(:));

if (error_dx > 1e-8) || (error_dw > 1e-8) || (error_db > 1e-8)
    fprintf('Matlab (FullyConnected BP) calculation is wrong\n');
else
    fprintf('Matlab (FullyConnected BP) calculation is right\n');
end


