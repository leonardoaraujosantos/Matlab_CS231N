function [ outMeans ] = getPixelMeanOnBatch( imgBatch )
% Reference:
% http://stackoverflow.com/questions/5689639/how-to-average-multiple-images-in-matlab
% Get the man value of every channel on the batch of images
% On our system an image batch is a 4d array with the following structure
% (Height)x(Width)x(channels)x(batchSize), example 10 images
% 320x240(Width x Height) will be 240x320x3x10, remember tha on
% matlab/numpy matrices are (rowsXcols)
nChannels = size(imgBatch,3);

% Calculate means, by the way this will work only if the batch is organized
% on the correct way, ie on CS231n solutions, the 4d array on numpy is
% represented like (batchSize)x(channels)x(Width)x(Height)
if nChannels == 3
    % RGB image
    allRedChannels = imgBatch(:,:,1,:);
    allGreenChannels = imgBatch(:,:,2,:);
    allBlueChannels = imgBatch(:,:,3,:);
    
    meanR = mean(allRedChannels(:));
    meanG = mean(allGreenChannels(:));
    meanB = mean(allBlueChannels(:));
    
    outMeans = {meanR,meanG,meanB};
else
    % Grayscale image
    intensityChannel = imgBatch(:,:,1,:);
    meanIntensity = mean(intensityChannel(:));
    outMeans = {meanIntensity};
end


end

