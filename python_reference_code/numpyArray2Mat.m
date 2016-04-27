function [ matArray ] = numpyArray2Mat( numpyArray )
%NUMPYARRAY2MAT Convert a numpy array to matlab array
% Example
% A = rand(2,3);
% numpyMatrix = matArray2Numpy(A)
% B = numpyArray2Mat(numpyMatrix);
% isequal(A,B)

%% Import numpy
py.importlib.import_module('numpy');

%% Get information about array
typeData = char(numpyArray.dtype.name);
shapeCell = cell(numpyArray.shape);
numDimensions = double(numpyArray.ndim);
sizeInfo = zeros(1,numDimensions);
for idx=1:numDimensions
    sizeInfo(idx) = double(shapeCell{idx});
end

%% Get the data
% Get double data from python numpy array
data = double(py.array.array('d', py.numpy.nditer(numpyArray, pyargs('order', 'F'))));
%data = double(py.array.array('d',py.numpy.nditer(pythonMatrix)));
matArray = reshape(data, sizeInfo);

end

