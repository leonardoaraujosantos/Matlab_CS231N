%% Convolution n dimensions
% The following code is just a extension of conv2d_vanila for n dimensions.
% Parameters:
% Input: H x W x depth
% K: kernel F x F x depth
% S: stride (How many pixels he window will slide on the input)
% This implementation is like the 'valid' parameter on normal convolution

function outConv = convn_vanilla(input, kernel, S)
% Get the input size in terms of rows and cols. The weights should have
% same depth as the input volume(image)
[rowsIn, colsIn, ~] = size(input);

% Get volume dimensio
depthInput = ndims(input);

% Get the kernel size, considering a square kernel always
F = size(kernel,1);

%% Initialize outputs
sizeRowsOut = ((rowsIn-F)/S) + 1;
sizeColsOut = ((colsIn-F)/S) + 1;
outConvAcc = zeros(sizeRowsOut , sizeColsOut, depthInput);

%% Do the convolution
% Convolve each channel on the input with it's respective kernel channel,
% at the end sum all the channel results.
for depth=1:depthInput
    % Select input and kernel current channel
    inputCurrDepth = input(:,:,depth);
    kernelCurrDepth = kernel(:,:,depth);
    % Iterate on every row and col, (using stride)
    for r=1:S:(rowsIn-1)
        for c=1:S:(colsIn-1)
            if (((c+F)-1) <= colsIn) && (((r+F)-1) <= rowsIn)
                % Select window on input volume
                sampleWindow = inputCurrDepth(r:(r+F)-1,c:(c+F)-1);
                % Do the dot product
                dotProd = sum(sampleWindow(:) .* kernelCurrDepth(:));
                % Store result
                outConvAcc(ceil(r/S),ceil(c/S),depth) = dotProd;
            end
        end
    end
end

% Sum elements over the input volume dimension (sum all the channels)
outConv = sum(outConvAcc,depthInput);
end