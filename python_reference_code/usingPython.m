%% Python on matlab tutorial
% During the development we will need to check for reference code
% written on python, so it's a good thing to learn how to connect custom
% python code on matlab.
%
% http://www.mathworks.com/help/matlab/matlab_external/undefined-variable-py-or-function-py-command.html#buialof-67
%
% http://www.mathworks.com/help/matlab/matlab_external/call-user-defined-custom-module.html
% 
% http://www.mathworks.com/help/matlab/matlab_external/use-python-list-of-numeric-types-in-matlab.html
%
% http://www.mathworks.com/help/matlab/matlab_external/limitations-to-python-support.html
%
% http://www.mathworks.com/help/matlab/matlab_external/call-modified-python-module.html
%
% http://www.mathworks.com/matlabcentral/answers/157347-convert-python-numpy-array-to-double
% 
% http://www.cert.org/flocon/2011/matlab-python-xref.pdf
%
% http://cs231n.github.io/python-numpy-tutorial/
%
% http://www.mathworks.com/help/matlab/matlab_external/handle-python-exceptions.html
%

% Point matlab to python executable
%pyversion 'C:\Anaconda2\python.exe'

% Load python and show  executable entry points
pyversion;

% To see if there is any problem loading a python module use:
py.importlib.import_module('simpleNumpy');

%% Create a python numpy matrix 2x3
pythonMatrix = py.simpleNumpy.createNumpyMatrix(2,3);

% Print python matrix
py.print(pythonMatrix);

%% Convert python matrix to matlab
% Get double data from python numpy array
data = double(py.array.array('d', py.numpy.nditer(pythonMatrix, pyargs('order', 'F'))));
pythonMatrixSize = pythonMatrix.shape;
matrixSize = [double(pythonMatrixSize{1}) double(pythonMatrixSize{2})];
%data = double(py.array.array('d',py.numpy.nditer(pythonMatrix)));
matlabMatrix = reshape(data, matrixSize);
disp(matlabMatrix);

%% Now send a matlab matrix 2d to python numpy
matMatrix = [1 2 3; 4 5 6];
disp(matMatrix)
% First we need to transpose the matrix to match numpy C array (row-major)
matMatrix = matMatrix';

% Matlab python only support transfering 1-N vectors (collumn vectors)
vec_1d_numpy = py.numpy.array(matMatrix(:)');
matPython = vec_1d.reshape(2,3);
py.print(matPython);