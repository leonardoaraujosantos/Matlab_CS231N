%% Accelerating convolution with im2col
% This example show how to implement convolution with matrix
% multiplication, the idea is take advantage of the matrix multiplication
% speed specially on hardware, to have faster convolutions, which are used
% a lot on convolutional neural networks. By the way the time that is loss
% during the forward propagation of convolutional neural networks is 90% on
% convolutions.
%
% <<../../docs/imgs/TimeConv.png>>
%
% The advantage of this method is described here:
%
% http://www.eecs.berkeley.edu/Pubs/TechRpts/2014/EECS-2014-93.pdf
%
% http://petewarden.com/2015/04/20/why-gemm-is-at-the-heart-of-deep-learning/
%
% http://arxiv.org/pdf/1501.07338v1.pdf
%
% http://knet.readthedocs.org/en/latest/cnn.html
%
% http://github.com/BVLC/caffe/blob/master/src/caffe/util/im2col.cpp
%
% http://cs.brown.edu/courses/cs143/results/proj1/jklee/
%
% http://yunmingzhang.wordpress.com/2015/03/13/how-does-3d-convolution-really-works-in-caffe-a-detailed-analysis/
%
% http://caffe.berkeleyvision.org/doxygen/index.html
%
% http://github.com/BVLC/caffe/issues/424
%

%% 2d Convolution
% Is a mathematical operation used to apply image filters. Is done by
% multiplying a pixel and it's neighboring pixels by a matrix, and this
% matrix keep sliding on the whole window.
%
% <<../../docs/imgs/animatedConv.gif>>
%
%
% <<../../docs/imgs/kernel_convolution.jpg>>
%
% When doing convolutions while you slide your kernel window, you will have
% cases where the kernel window does not fit on the image, on this cases
% you decide if you completely ignore those values and only accept the
% calculation when the kernel window is fully inserted on the image (valid)
% or you ignore those out values (replace by zeros), or pad the image with
% zeros (same).
%
% It should be noted that the matrix operation being performed - 
% convolution - is not traditional matrix multiplication, despite being 
% similarly denoted by *. In words convolution is: given two three-by-three
% matrices, one a kernel, and the other an image piece, convolution is the 
% process of multiplying locationaly similar entries and summing
%
% <<../../docs/imgs/conv_full_same_valid.gif>>
%
% http://stackoverflow.com/questions/14864315/alternative-to-conv2-in-matlab
%
% http://web.pdx.edu/~jduh/courses/Archive/geog481w07/Students/Ludwig_ImageConvolution.pdf
%
% http://www.songho.ca/dsp/convolution/convolution2d_example.html
%

%% Non-Vectorized implementation
% Basically we will compare the execution time with this 2d convolution
% implementation, with the vectorized implementation and the conv2 for
% grayscale images and convn for color images
%
% <<../../docs/imgs/ConvAlgo.png>>
%
% Code:
%
% <include>convolve2d.m</include>
%


%% The Idea
% The idea is to transform the input image into a set of collumn vectors
% that will then be multiplied by the kernel
%
% <<../../docs/imgs/im2col_1.png>>
%
% <<../../docs/imgs/im2col_3.png>>
%
% Matlab does have a implementation for im2col, but it lacks of the
% following capabilities that will be necessary later for implementing
% other layers (like pool)
%%
% 
% * Does not allows padding
% * Preserves the spatial layout of the local patches
% * Don't support strides (needed if you have different types of conv
% layers)
%
% Some implementations examples (Python, C++, Cuda, Matlab)
%%
% 
% * http://github.com/tsogkas/utils/blob/master/im2patches.m
% * http://github.com/MyHumbleSelf/cs231n/blob/master/assignment2/cs231n/im2col.py
% * http://github.com/BVLC/caffe/blob/master/src/caffe/util/im2col.cpp
% * http://github.com/BVLC/caffe/blob/master/src/caffe/util/im2col.cu
% * http://stackoverflow.com/questions/30109068/implement-matlabs-im2col-sliding-in-python
% * http://github.com/samehkhamis/pydeeplearn/blob/master/pydeeplearn/image/image.pyx
% * http://github.com/fmassa/torch-nn/blob/master/ConvLua/im2col.c
% * http://github.com/yjxiong/im2col
% 



%% Simple example
% To show the process we will do the operations needed to do a convolution
% with matrix multiplication like the picture bellow
%
% <<../../docs/imgs/im2col2.jpg>>
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
X_col = im2col(X,size(W))

%%
%
% Now to convolve just multiply and reshape back the results
result_im2col_conv = reshape(W_col * X_col, size_valid_conv)

%% Benchmarking with real life example
% Here we compare the time spent doing a convolution with the operations
% needed to convolve with matrix multiplication(im2col) observe that the
% time to execute is even worse with im2col
% We will convolve the image with the Gx sobel kernel
%

% Loading image
imgCat = imread('datasets/imgs/CatImg.png');
Gx = [-1 0 1; -2 0 2; -1 0 1];
imshow(imgCat);

%%
%
% Convolving with conv2 (if is color should be convn)
imgCat = double(imgCat);
tic; imgResult = conv2(imgCat,Gx,'valid'); timeSpent = toc;
sizeResult = size(imgResult);
imshow(imgResult);
fprintf('Took %d seconds to complete(conv2)\n',timeSpent);

%%
%
% Now with im2col
tic;
W = flipud(fliplr(Gx));
W_col = W(:)';
X_col = im2col(imgCat,size(W));
resConvIm2col = W_col * X_col;
timeSpent = toc;
result_im2col_conv = reshape(resConvIm2col, sizeResult);
imshow(result_im2col_conv);
fprintf('Took %d seconds to complete (im2col)\n',timeSpent);

%%
%
% Convolving with vanilla convolution (non-vectorized)
imgCat = double(imgCat);
tic; imgResult = convolve2d(imgCat,Gx); timeSpent = toc;
imshow(imgResult);
fprintf('Took %d seconds to complete(non-vec conv2)\n',timeSpent);

%% Now emulating a real life convnet.
% The vectorization of the convolutional neural network will shine on this
% case were for instance, a single image will be convolved 512 times (ie
% First layer of alexnet)
%
% <<../../docs/imgs/ConvFilters.png>>
%
%
% <<../../docs/imgs/ConvFilters_2.png>>
%
% Here C1 is the output of 512 convolutions of the same image with 512
% different kernels, so we do just one im2col(slow) operations instead of
% 512 complete convolution loops.
%

% Loading image
imgCat = imread('datasets/imgs/CatImg.png');
Gx = [-1 0 1; -2 0 2; -1 0 1];
imshow(imgCat);

%%
%
% With normal convolution
imgCat = double(imgCat);
tic;
for idxConv=1:512
    %imgResult = conv2(imgCat,Gx,'valid');
    imgResult = convolve2d(imgCat,Gx);
end
timeSpentNonVec = toc;
imshow(imgResult);
fprintf('Took %d seconds to complete 512 non-vectorized convs\n',timeSpentNonVec);

%%
%
% With vectorization no-GPU
W = flipud(fliplr(Gx));
W_col = W(:)';
X_col = im2col(imgCat,size(W));
tic;
for idxConv=1:512
    resConvIm2col = W_col * X_col;
end
timeSpent = toc;
result_im2col_conv = reshape(resConvIm2col, sizeResult);
imshow(result_im2col_conv);
diffTime = timeSpentNonVec / timeSpent;
fprintf('Took %d seconds to complete 512 vectorized convs grayscale, speedup=%dx\n',timeSpent,round(diffTime));

%%
%
% With vectorization GPU

W = flipud(fliplr(Gx));
W_col = W(:)';
X_col = im2col(imgCat,size(W));
gpuArray(single(W_col));
gpuArray(single(X_col));
tic;
for idxConv=1:512
    resConvIm2col = W_col * X_col;
end
timeSpent = toc;
result_im2col_conv = reshape(resConvIm2col, sizeResult);
imshow(result_im2col_conv);
diffTime = timeSpentNonVec / timeSpent;
fprintf('Took %d seconds to complete 512 vectorized convs(GPU) grayscale, speedup=%dx\n',timeSpent,round(diffTime));

%% Handle colors
% With color images, we need to apply the matrix multiplication for every
% channel, on this case the im2col is even a little bit faster than convn
% from Matlab.
%
% By the way if we're doing convolutions with color images, we are doing
% actually 3d convolutions and the "convolve2d" code given earlier will not
% work (TODO: convolve3d)

% Loading image
imgCat = imread('datasets/imgs/catColor.jpg');
Gx = [-1 0 1; -2 0 2; -1 0 1];
imshow(imgCat);

%%
%
% With convn
imgCat = double(imgCat);
tic;
for idxConv=1:512
    imgResult = convn(imgCat,Gx,'valid');
end
timeSpentConvn = toc;
sizeResult = size(imgResult);
imshow(imgResult);
fprintf('Took %d seconds to complete 512 (convn CPU) color\n',timeSpentConvn);

%%
%
% With vectorization

W = flipud(fliplr(Gx));
W_col = W(:)';
X_col_R = im2col(imgCat(:,:,1),size(W));
X_col_G = im2col(imgCat(:,:,2),size(W));
X_col_B = im2col(imgCat(:,:,3),size(W));
resConvIm2col = zeros(size(imgCat));

W_col = gpuArray(single(W(:)'));
X_col_R = gpuArray(single(X_col_R));
X_col_G = gpuArray(single(X_col_G));
X_col_B = gpuArray(single(X_col_B));

tic;
for idxConv=1:512
    prod_R = W_col * X_col_R;
    prod_G = W_col * X_col_G;
    prod_B = W_col * X_col_B;
end
result_im2col_conv = reshape([prod_R prod_G prod_B], sizeResult);
timeSpent = toc;
result_im2col_conv = gather(result_im2col_conv);
imshow(result_im2col_conv);
diffTime = timeSpentConvn / timeSpent;
fprintf('Took %d seconds to complete 512 vectorized convs(GPU) color, speedup=%dx\n',timeSpent,round(diffTime));
