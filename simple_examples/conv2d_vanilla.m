%% Convolution 2d
% The following code will implement a simple 2d convolution algorithm.

%% Mathematical formula, discrete 1d
% 
% <<../../docs/imgs/matFormulaDiscrete1d.PNG>>
% 
%% Algorithm high level description 1d
% 
% <<../../docs/imgs/Algorithm_C_1d.PNG>>
% 
%% Effect wanted, digital filtering
% 
% <<../../docs/imgs/Convolution_of_box_signal_with_itself2.gif>>
% 
%% Some sample of 1d filters
% 
% <<../../docs/imgs/1dFilters.png>>
%
%%
% 
% <<../../docs/imgs/1dFilterEffects.gif>>
% 

%% Mathematical formula, discrete 2d
% 
% <<../../docs/imgs/matFormulaDiscrete2d.PNG>>
% 

%% Algorithm high level description 2d
% 
% <<../../docs/imgs/AlgorithmHighLevel.PNG>>
% 

%% Kernel and Input signal operation
% 
% <<../../docs/imgs/kernel_convolution.jpg>>
% 

%% Ilustration
% 
% <<../../docs/imgs/SampleConvolution.PNG>>
% 
%% Types of 2d filters and it's effects
% 
% <<../../docs/imgs/2dFilters.png>>
% 
% <<../../docs/imgs/Effects_2d.png>>
%

%% Correlation
% Another use for the convolution is to look for a certain pattern on
% images, for instance imagine that you want to look for George washington
% on some image
% 
% <<../../docs/imgs/PatternMatch.png>>
% 
% The first thing that you need to do is to rotate 180º the target, then
% convolve the target with the image that you want to look for, sometimes
% you can also do some kind of filtering on the target (ie: sobel)
% 
% <<../../docs/imgs/PatternMatch2.png>>
% 
% <<../../docs/imgs/PatternMatch4.png>>
% 
% <<../../docs/imgs/PatternMatch3.png>>
%


%% Code explanation
% As seen on the function interface, it has the input(image) and kernel,
% the first thing that the algorithm will do is to infer the input sizes,
% pad the original image with zeros to garantee the calculation on
% off-border cases.
% Then a loop will iterate through every pixel and a inner loop will
% populate a window that will be used to calculate the convolution
function outConv = conv2d_vanila(input, kernel)
%% Get the input size in terms of rows and cols
[rowsIn, colsIn] = size(input);

%% Initialize outputs to have the same size of the input
outConv = zeros(rowsIn , colsIn);

%% Get kernel size
[rowsKernel, colsKernel] = size(kernel);

%% Initialize a sampling window
window = zeros(rowsKernel , colsKernel);

%% Rotate the kernel 180º
rot_kernel = rot90(kernel, 2);

%% Calculate the number of elements to pad
num2Pad = floor(rowsKernel/2);
paddedInput = padarray(input,[num2Pad num2Pad]);

%% Sample the input signal to form the window
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
        outConv(idxRowsIn, idxColsIn) = doConvolution(window,rot_kernel);
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

%% Examples
% 
%   input = [1 2 3; 4 5 6; 7 8 9]
%   kernel = [1 2 1; 0 0 0; -1 -2 -1]
%   conv2d_vanila(input,kernel)
%   conv2(input,kernel,'same')
%   imageInput = imread('OldGlory.bmp');
%   grad_7x7 = [0 0 -1 -1 -1 0 0;0 0 -1 1 -1 0 0;-1 -1 2 3 2 -1 -1;-1 1 3 -4 3 1 -1;-1 -1 2 3 2 -1 -1;0 0 -1 1 -1 0 0;0 0 -1 1 -1 0 0];
%   result_manual = conv2d_manual(double(imageInput),grad_7x7);
% 