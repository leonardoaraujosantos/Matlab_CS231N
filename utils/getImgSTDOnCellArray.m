function [ stdOut ] = getImgSTDOnCellArray( imgCellArray )
% Get the man value of every channel on the batch of images
numImagesBatch = numel(imgCellArray);
stdOut = zeros(1,3);

stdRed = zeros(1,numImagesBatch);
stdGreen = zeros(1,numImagesBatch);
stdBlue = zeros(1,numImagesBatch);

for idxImg = 1:numImagesBatch
    img = imgCellArray{idxImg};
    [height_rows,width_cols,numChannels] = size(img);
    if (numChannels == 3)
        % Color image
        RedChannel = img(:,:,1);
        GreenChannel = img(:,:,2);
        BlueChannel = img(:,:,3);

        stdRed(idxImg) = std2(RedChannel);
        stdGreen(idxImg) = std2(GreenChannel);
        stdBlue(idxImg) = std2(BlueChannel);
    else
        % Grayscale image        
        stdIntensity = std2(img);
        stdRed(idxImg) = stdIntensity;
        stdGreen(idxImg) = stdIntensity;
        stdBlue(idxImg) = stdIntensity;
    end     
end

std_R_val = sum(stdRed) / numImagesBatch;
std_G_val = sum(stdGreen) / numImagesBatch;
std_B_val = sum(stdBlue) / numImagesBatch;
stdOut = {std_R_val,std_G_val,std_B_val};

end

