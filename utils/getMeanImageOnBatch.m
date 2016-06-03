function [ meanImage ] = getMeanImageOnBatch( imgBatch )
% Generate a mean image with all images on the batch
% On our system an image batch is a 4d array with the following structure
% (Height)x(Width)x(channels)x(batchSize), example 10 images
% 320x240(Width x Height) will be 240x320x3x10, remember tha on
% matlab/numpy matrices are (rowsXcols)
dimsBatch = size(imgBatch);
numImagesBatch = dimsBatch(4);
meanImage = (zeros(dimsBatch(1:3)));

% Check if numImagesBatch could explode the number of bits choosen to
% accumulate the image
maxBatchSize = (2^64)/255;
if numImagesBatch > maxBatchSize
    disp('Batch to big to compute');
    return;
end

for idxImg = 1:numImagesBatch
    img = (imgBatch(:,:,:,idxImg));
    meanImage = imadd(img,meanImage);
end

meanImage = meanImage / numImagesBatch;

end

