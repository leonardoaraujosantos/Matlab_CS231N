function [ images_mem ] = loadImages( imageFilenames )
%LOADIMAGESDIRECTORY Create a cell array with all images on imageFilenames
% Example:
% files = getFilenamesFromDirectory('/mnt/fs3/QA_analitics/Apical_CNN_training_data/ResNetPedestrianDataset/train');
%

numImgsFile = numel(imageFilenames);

for idxImg = 1:numImgsFile
    try
        images_mem{idxImg} = imread(imageFilenames{idxImg});
    catch ME
        fprintf('Error loading image "%s" at index:%d\n',imageFilenames{idxImg},idxImg);
    end
end
end

