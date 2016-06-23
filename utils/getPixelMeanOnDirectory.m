function [ outMeans ] = getPixelMeanOnDirectory( imgDirectory)
% Get the man value of every channel on directory
% example: meanPixelData = getPixelMeanOnDirectory('/mnt/fs3/QA_analitics/Apical_CNN_training_data/ResNetPedestrianDataset/train/pedestrian_images');

%% Load image filenames
fprintf('Loading filenames from %s...\n',imgDirectory);
tic;
imgFileNames = getFilenamesFromDirectory(imgDirectory);
mapTime = toc;

totalNumImg = numel(imgFileNames);
fprintf('Total number of images %d...\n',totalNumImg);
fprintf('Time for local machine to map all files: %d seconds\n',mapTime);


%% Estimate how much data each worker will handle
% Inspect system capabilities
systemCaps = SystemCapabilitiesExplorer();
% Get number of available workers on the current pool
numWorkers = systemCaps.getNumCurrentWorkers();
fprintf('Number of available workers on cluster: %d\n',numWorkers);

% Divide the images on the cluster
numImagesPerCluster = floor(totalNumImg / numWorkers);

%% Distributing computing on cluster
% Verify the the time to load the data is 100x slower than actually
% calculating the mean value.
fprintf('Distributing on <%d clusters> %d images each cluster handle <%d> images\n',numWorkers, totalNumImg, numImagesPerCluster);
spmd
    splitFilenames = splitCells(imgFileNames,numWorkers);
end

% The idea of spmd (single program multiple data) is that the data that we
% will use is referenced by the variable "labindex" will be different for
% each cluster.
% So the code between spmd..end will work on every cluster, but the data in
% which it will work on depend on the cluster number (labindex)
% All console-output functions ex: fprintf will be displayed on the local
% machine
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

% Gather back results of each worker (stored on "meanPixel")
outMeans = sum(cell2mat(meanPixel(:)))/numWorkers;

