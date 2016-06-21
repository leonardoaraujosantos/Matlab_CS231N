function [ outMeans ] = getPixelMeanOnDirectory( imgDirectory)
% Get the man value of every channel on directory
% example: meanPixelData = getPixelMeanOnDirectory('/mnt/fs3/QA_analitics/Apical_CNN_training_data/ResNetPedestrianDataset/train/pedestrian_images');

%% Load image filenames
fprintf('Loading filenames from %s...\n',imgDirectory);
imgFileNames = getFilenamesFromDirectory(imgDirectory);

totalNumImg = numel(imgFileNames);
fprintf('Total number of images %d...\n',totalNumImg);

%% Estimate amount of memory needed to store images
% Inspect system capabilities
systemCaps = SystemCapabilitiesExplorer();
%freeMem = systemCaps.getMemAvailable();
freeMem = 80;
freeMem = freeMem * 10e6;
% Get up to 30% of the free memory available
freeMem = freeMem * 0.3;

% Get first image
firstImg = imread(imgFileNames{ceil(totalNumImg/2)});
sizeImage = size(firstImg);
sizeImageBytes = prod(sizeImage);

% Maximum number of images without exploding memory (30%)
maxNumImages = floor(freeMem / sizeImageBytes);
if totalNumImg < maxNumImages
    maxNumImages = totalNumImg;
end

%% Estimate how much data each worker will handle
% Get number of available workers on the current pool
numWorkers = systemCaps.getNumCurrentWorkers();
fprintf('Number of available workers on cluster: %d\n',numWorkers);

% Divide the images on the cluster
numImagesPerCluster = floor(totalNumImg / numWorkers);

%% Distributing on cluster
fprintf('Distributing on <%d clusters> %d images each cluster handle <%d> images\n',numWorkers, totalNumImg, numImagesPerCluster);
spmd
    splitFilenames = splitCells(imgFileNames,numWorkers);
end

spmd
    tic;
    imgsMem = loadImages(splitFilenames{labindex});
    timeLoadData = toc;
    fprintf('Worker %d finished to load data in %d seconds\n',labindex,timeLoadData);
    tic;
    meanPixel = cell2mat(getPixelMeanOnCellArray(imgsMem));
    timeMean = toc;
    fprintf('Worker %d finished mean calculation in %d seconds\n',labindex,timeMean);
end
outMeans = sum(cell2mat(meanPixel(:)))/numWorkers;

