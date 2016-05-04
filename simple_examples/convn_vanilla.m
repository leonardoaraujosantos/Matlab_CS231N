%% Convolution n dimensions
% The following code is just a extension of conv2d_vanila for n dimensions.
% Parameters:
% K: kernel
% S: stride

function outConv = convn_vanila(input, kernel, S)
%% Get the input size in terms of rows and cols
[rowsIn, colsIn, numDims] = size(input);

% Get the kernel size, considering a square kernel always
F = size(kernel,1);

%% Initialize outputs
sizeRowsOut = ((rowsIn-F)/S) + 1;
sizeColsOut = ((colsIn-F)/S) + 1;
outConv = zeros(sizeRowsOut , sizeColsOut, numDims);


%% Initialize a sampling window
window = zeros(F , F);

%% Rotate the kernel 180�
rot_kernel = rot90(kernel, 2);

%% Sample the input signal to form the window
% Iterate on every dimension  
for idxDims=1:ndims(input)
    inputCurDim = input(:,:,idxDims);
    % Iterate on every element of the input signal
    % Iterate on every row
    for idxRowsIn = 1 : rowsIn
        % Iterate on every col
        for idxColsIn = 1 : colsIn
            % Populate our window (same size of the kernel)
            for idxRowsKernel = 1 : F
                for idxColsKernel = 1 : F                    
                    % Slide the window
                    slideRow = (idxRowsIn-1)*S;
                    slideCol = (idxColsIn-1)*S;
                    
                    % Sample the window, but avoid going out of the input
                    if (idxRowsKernel + slideRow) <= size(inputCurDim,1) && idxColsKernel + slideCol <= size(inputCurDim,2)
                        window(idxRowsKernel,idxColsKernel) = ...
                            inputCurDim(idxRowsKernel + slideRow, idxColsKernel + slideCol);
                    end
                end
            end
            % Calculate the convolution here...
            outConv(idxRowsIn, idxColsIn,idxDims) = doConvolution(window,rot_kernel);
        end
    end
end

%% Moving window effect
% The previous inner for loop updates the variables slideRow, and slideCol
% those updates will create the following effect
%
% <<../../docs/imgs/3D_Convolution_Animation.gif>>
%

end

%% Calculate the sum of the product between the kernel and the window
% The convolution is all about the sum of product of the window and kernel,
% bye the way this is a dot product
function result = doConvolution(window, kernel)
result = sum(sum(window .* kernel));
end
