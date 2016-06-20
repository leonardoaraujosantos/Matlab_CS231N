function [ images_mem ] = loadImages( imageFilenames )
%LOADIMAGESDIRECTORY Create a cell array with all images on imageFilenames
% Example:
% files = getFilenamesFromDirectory('/mnt/fs3/QA_analitics/Apical_CNN_training_data/ResNetPedestrianDataset/train');
%

numImgsFile = numel(imageFilenames);

% Inspect system capabilities
systemCaps = SystemCapabilitiesExplorer();
freeMem = systemCaps.getMemAvailable();
freeMem = freeMem * 10e6;

% Get up to 30% of the free memory available
freeMem = freeMem * 0.3;

% Get first image
firstImg = imread(imageFilenames{1});
sizeImage = size(firstImg);
sizeImageBytes = prod(sizeImage);

maxNumImages = freeMem / sizeImageBytes;

% Get number of available workers on the current pool
numWorkers = systemCaps.getNumCurrentWorkers();
fprintf('Number of available workers on cluster: %d\n',numWorkers);

% Divide the images on the cluster
imgsPerCluster = numImgsFile / numWorkers;

if numImgsFile < maxNumImages
    parfor idxImg = 1:numImgsFile
        try
            images_mem{idxImg} = imread(imageFilenames{idxImg});
        catch ME
            fprintf('Error loading image "%s" at index:%d\n',imageFilenames{idxImg},idxImg);
        end
    end
else
    disp('Not enough memory to hold images at once');
    numIterationsPerMemBlock = floor(numImgsFile / maxNumImages);
    idxCountImg = int32(1);
    for idxNumIterMemBlock=1:numIterationsPerMemBlock        
        fprintf('Doing block %d/%d\n',idxNumIterMemBlock,numIterationsPerMemBlock);
        limitEndLoop = int32(idxCountImg + floor(maxNumImages));
        parfor idxImg = idxCountImg:limitEndLoop
            try
                images_mem{idxImg} = imread(imageFilenames{idxImg});
            catch ME
                fprintf('Error loading image "%s" at index:%d\n',imageFilenames{idxImg},idxImg);
            end
        end
        idxCountImg = int32(idxCountImg) + int32(maxNumImages);
    end
end

end

