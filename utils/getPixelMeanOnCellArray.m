function [ outMeans ] = getPixelMeanOnCellArray( imgCellArray )
% Get the man value of every channel on the batch of images
numImagesBatch = numel(imgCellArray);
meanImage = zeros(1,3);

meanR = zeros(1,numImagesBatch);
meanG = zeros(1,numImagesBatch);
meanB = zeros(1,numImagesBatch);
for idxImg = 1:numImagesBatch
    img = imgCellArray{idxImg};
    [height_rows,width_cols,numChannels] = size(img);
    if (numChannels == 3)
        % Color image
        allRedChannels = img(:,:,1);
        allGreenChannels = img(:,:,2);
        allBlueChannels = img(:,:,3);

        meanR(idxImg) = mean(allRedChannels(:));
        meanG(idxImg) = mean(allGreenChannels(:));
        meanB(idxImg) = mean(allBlueChannels(:));
    else
        % Grayscale image        
        meanIntensity = mean(img(:));
        meanR(idxImg) = meanIntensity;
        meanG(idxImg) = meanIntensity;
        meanB(idxImg) = meanIntensity;
    end    
end

meanR_val = sum(meanR(:)) / numImagesBatch;
meanG_val = sum(meanG(:)) / numImagesBatch;
meanB_val = sum(meanB(:)) / numImagesBatch;
outMeans = {meanR_val,meanG_val,meanB_val};

end

