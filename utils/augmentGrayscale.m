function [ timeToConvert ] = augmentGrayscale( input_file )
%augmentGrayscale Get all images on file and create a grayscale version
% Ex:
% augmentGrayscale('/mnt/fs3/QA_analitics/For_Leo/Train.txt')

% Get the number of lines on the file
numlines = 0;
if (isunix) %# Linux, mac
    [~, result] = system( ['wc -l ', input_file] );
    result = strsplit(result);
    numlines = str2num(result{1});
    
elseif (ispc) %# Windows
    % TODO
    % type c:\windows\win.ini | find /c /v "~~~"
    numlines = 0;    
end

fprintf('This file has %d lines/samples\n',numlines);
fInputFileHandler = fopen(input_file,'r');
lineFile = fgetl(fInputFileHandler);
tic;

% Parse the file contents, and convert the filename on each line 
contFiles = 1;
while ischar(lineFile)
    % Get first collumn
    cellStrSplit = strsplit(lineFile);
    fileNameSample = cellStrSplit{1};
    
    % Open image
    try
        img = imread(fileNameSample);
        img_grayed = img;
    catch
       fprintf('Open file %s failed\n', fileNameSample);
       lineFile = fgetl(fInputFileHandler);
       continue;
    end
    
    % Select channels
    R = img(:,:,1);
    G = img(:,:,2);
    B = img(:,:,3);
    % Convert to grayscale but keep the number of channels.
    img_grayed(:,:,1) = 0.2989 * R + 0.5870 * G + 0.1140 * B;
    img_grayed(:,:,2) = img_grayed(:,:,1);
    img_grayed(:,:,3) = img_grayed(:,:,1);
    
    % Write on disk (same name, but with "_gray")
    [pathstr,name,ext] = fileparts(fileNameSample);
    name = [name '_gray'];
    fileNameSample_new = [pathstr filesep name ext];
    imwrite(img, fileNameSample_new);
    
    % Just display progress....
    contFiles = contFiles + 1;
    if mod(contFiles,1000) == 0
       fprintf('Processing files %d/%d\n',contFiles,numlines);
    end
    
    % Next line
    lineFile = fgetl(fInputFileHandler);
end
timeToConvert = toc;
% Close file
fclose(fInputFileHandler);
end

