%% Accelerating convolution with im2col
% This example show how to implement convolution with matrix
% multiplication, the idea is take advantage of the matrix multiplication
% speed specially on hardware, to have faster convolutions, which are used
% a lot on convolutional neural networks. By the way the time that is loss
% during the forward propagation of convolutional neural networks is 90% on
% convolutions.
%
% <</home/leo/work/Matlab_CS231N/docs/imgs/TimeConv.png>>
%
% The advantage of this method is described here:
%
% http://www.eecs.berkeley.edu/Pubs/TechRpts/2014/EECS-2014-93.pdf
%
% http://petewarden.com/2015/04/20/why-gemm-is-at-the-heart-of-deep-learning/
%
% http://arxiv.org/pdf/1501.07338v1.pdf
%

%% The Idea
% The idea is to transform the input image into a set of collumn vectors
% that will then be multiplied by the kernel
%
% <</home/leo/work/Matlab_CS231N/docs/imgs/im2col_1.png>>
%
% <</home/leo/work/Matlab_CS231N/docs/imgs/im2col_3.png>>
%

%% Simple example
% Starting with a simple example, to understand the better this stuff...
%
% <</home/leo/work/Matlab_CS231N/docs/imgs/im2col2.jpg>>
%

% Defining the data
W = [1 3; 2 4]
X = [1 4 7; 2 5 8; 3 6 9]


%%
%
% Doing the normal convolution using the conv2 from matlab, which is fast
% already, here we also get the result size that will be used later.
reference_result = conv2(X,W,'valid')
% No need to do a convolution to know this size ok.
size_valid_conv = size(reference_result);

%%
%
% Prepare the kernel matrix (W)
W = flipud(fliplr(W))
W_col = W(:)'

%%
%
% Transform input X on collumns (im2col)
X_col = im2col(X,[size(W,1) size(W,2)])

%%
%
% Now to convolve just multiply and reshape back the results
result_im2col_conv = reshape(W_col * X_col, size_valid_conv)

%% Benchmarking with real life example
% Now we will use a RGB image with various sizes and measure the time spent
% to calculate it's convolution.
% We will convolve the image with the Gx sobel kernel
%

% Loading image
imgCat = imread('/home/leo/work/Matlab_CS231N/docs/imgs/CatImg.jpg');
Gx = [-1 0 1; -2 0 2; -1 0 1];
imgri
imshow(imgCat);

%%
%
% Convolving RGB (3d) image with imfilter
tic; imgResult = imfilter(imgCat,Gx,'conv'); timeSpent = toc;
imshow(imgResult);
fprintf('Took %d seconds to complete\n',timeSpent);



