function [ timeToConvert ] = resizeImages( input_file, newSize )
%RESIZEIMAGES Resize all images pointed on the text file "input_file" to a
% "newSize"
% Ex:
% resizeImages('/mnt/fs3/QA_analitics/For_Leo/Train.txt', [128 128])

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
while ischar(lineFile)
    % Get first collumn
    cellStrSplit = strsplit(lineFile);
    fileNameSample = cellStrSplit{1};
    
    % Open image
    img = imread(fileNameSample);
    % Resize
    img = imresize(img, newSize);
    % Write on disk (same name)
    imwrite(img, fileNameSample);
    
    % Next line
    lineFile = fgetl(fInputFileHandler);
end
timeToConvert = toc;
% Close file
fclose(fInputFileHandler);
end

