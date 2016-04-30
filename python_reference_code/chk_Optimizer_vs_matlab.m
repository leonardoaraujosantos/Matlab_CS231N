%% Test Python version vs Matlab (Optimizer)
% On this script we will test the optimizer class, which is responsible to
% update the model parameters


%% Python preparation
% Add on python path assigment 2 (With teacher softmax)
% On the cs231n assigments 2 directory
clear all; clc;
insert(py.sys.path,int32(0),[pwd filesep 'python_reference_code' ...
    filesep 'cs231n_2016_solutions' filesep ...
    'assignment2' filesep 'cs231n']);
py.importlib.import_module('optim');

%% Define data
load optimResults

%% Call python reference code (sgd_momentum)
configPython = config;
configPython.velocity = matArray2Numpy(v);
python_optim_res = cell(py.optim.sgd_momentum(matArray2Numpy(w),matArray2Numpy(dw),configPython));
pythonDw = numpyArray2Mat(python_optim_res{1});
% A python dictionary is converted to a matlab struct
pythonConfigs = struct(python_optim_res{2});
pythonConfigs.velocity = numpyArray2Mat(pythonConfigs.velocity);
% 
% % Just display
fprintf('External Python CS231n sgd_momentum expected values\n');
disp('expected_new_dw');
disp(expected_next_w);
disp('Velocity');
disp(expected_velocity);

% % Just display
fprintf('External Python CS231n sgd_momentum calculated externally\n');
disp('dw');
disp(pythonDw);
disp('Velocity');
disp(pythonConfigs.velocity);

%% Display reference values from previous python simulation on exercise
fprintf('External Python CS231n sgd_momentum calculated externally\n');

%% Executing matlab version
optMat = Optimizer();
optMat.configs = configPython;
optMat.configs.velocity = v;
newDwMatlab = optMat.sgd_momentum(w,dw);

% % Just display
fprintf('Matlab sgd_momentum calculated externally\n');
disp('dw');
disp(newDwMatlab);
disp('Velocity');
disp(optMat.configs.velocity);
