function [ MatrixOut ] = padMatrixZeros( MatrixIn, numPad )
% Pad input matrix with zeros, if you have a matrix YxY and pad with 1, it
% will become (Y+1*2)x(Y+1*2)

% Allocate MatrixOut
sizeMatrixIn = size(MatrixIn);

% Alocate output to avoid realocation
if isa(MatrixIn,'gpuArray')
    MatrixOut = gpuArray(zeros(sizeMatrixIn(1)+(numPad*2),sizeMatrixIn(2)+(numPad*2)));
else
    MatrixOut = zeros(sizeMatrixIn(1)+(numPad*2),sizeMatrixIn(2)+(numPad*2));
end

% Insert MatrixIn on MatrixOut (Should be on the right spot)
MatrixOut((numPad+1):end-numPad,(numPad+1):end-numPad) = MatrixIn;

end

