function [ numpyArray ] = matArray2Numpy( matArray )
%MATARRAY2NUMPY Convert a matlab array to numpy
% Example
% A = rand(2,3);
% numpyMatrix = matArray2Numpy(A)


%% Import numpy
py.importlib.import_module('numpy');

%% Get information about matrix
numDimensions = ndims(matArray);
sizeInfo = size(matArray);
typeData = class(matArray);
scratchData = matArray;

%% Transform data from colmajor to rowmajor (for every dimension)
if numDimensions > 1    
    if numDimensions == 3
        for idxDims=1:numDimensions
            scratchData(:,:,idxDims) = scratchData(:,:,idxDims)';
        end
    elseif numDimensions == 2
        scratchData = scratchData';
    end
end

%% Create a numpy array
% Matlab python only support transfering 1-N vectors (collumn vectors)
vec_1d_numpy = py.numpy.array(scratchData(:)');
if numDimensions > 2
    numpyArray = vec_1d_numpy.reshape(fliplr(sizeInfo));
else
    numpyArray = vec_1d_numpy.reshape(sizeInfo);
end

%% Convert to correct type
numpyArray = numpyArray.astype(typeData);
% if isequal(typeData,'uint8')
%     numpyArray = numpyArray.astype(typeData);
% end
% if isequal(typeData,'int64')
%     numpyArray = numpyArray.astype(typeData);
% end

end

