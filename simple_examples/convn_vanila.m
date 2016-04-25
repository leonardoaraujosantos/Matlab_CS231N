%% Convolution n dimensions
% The following code is just a extension of conv2d_vanila for n dimensions.

function outConv = convn_vanila(input, kernel)
%% Get the input size in terms of rows and cols
[rowsIn, colsIn, numDims] = size(input);

%% Initialize outputs to have the same size of the input
outConv = zeros(rowsIn , colsIn, numDims);

%% Get kernel size
[rowsKernel, colsKernel] = size(kernel);

%% Initialize a sampling window
window = zeros(rowsKernel , colsKernel);

%% Rotate the kernel 180º
rot_kernel = rot90(kernel, 2);

%% Calculate the number of elements to pad
num2Pad = floor(rowsKernel/2);

%% Sample the input signal to form the window
for idxDims=1:ndims(input)
    paddedInput = padarray(input(:,:,idxDims),[num2Pad num2Pad]);
    % Iterate on every element of the input signal
    for idxRowsIn = 1 : rowsIn
        for idxColsIn = 1 : colsIn
            % Populate our window (same size of the kernel)
            for idxRowsKernel = 1 : rowsKernel
                for idxColsKernel = 1 : colsKernel
                    % Slide the window
                    slideRow = idxRowsIn - 1;
                    slideCol = idxColsIn -1;
                    
                    % Sample the window
                    window(idxRowsKernel,idxColsKernel) = ...
                        paddedInput(idxRowsKernel + slideRow, idxColsKernel + slideCol);
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
