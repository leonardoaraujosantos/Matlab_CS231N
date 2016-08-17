function [ timeToConvert ] = augmentDataSet( input_file )
%augmentDataSet Get all images on file and augment them.
% Ex:
% augmentDataSet('/mnt/fs3/QA_analitics/For_Leo/Train.txt')
augment = TransformBatch();

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
    catch
       fprintf('Open file %s failed\n', fileNameSample);
       lineFile = fgetl(fInputFileHandler);
       continue;
    end
    
    %% Grayscale
    img_augmented = augment.convertToGrayscale(img);
    
    % Write on disk (same name, but with "_gray")
    [pathstr,name,ext] = fileparts(fileNameSample);
    name = [name '_gray'];
    fileNameSample_new = [pathstr filesep name ext];
    imwrite(img_augmented, fileNameSample_new);
    
    %% Sepia
    img_augmented = augment.sepiaFilter(img);
    
    % Write on disk (same name, but with "_sepia")
    [pathstr,name,ext] = fileparts(fileNameSample);
    name = [name '_sepia'];
    fileNameSample_new = [pathstr filesep name ext];
    imwrite(img_augmented, fileNameSample_new);
    
    %% Flip color
    img_augmented = augment.flip_Color_Image(img);
    
    % Write on disk (same name, but with "_flipcolor")
    [pathstr,name,ext] = fileparts(fileNameSample);
    name = [name '_flipcolor'];
    fileNameSample_new = [pathstr filesep name ext];
    imwrite(img_augmented, fileNameSample_new);
    
    %% Add noise
    img_augmented = augment.addPeperNoise(img);
    
    % Write on disk (same name, but with "_noise")
    [pathstr,name,ext] = fileparts(fileNameSample);
    name = [name '_noise'];
    fileNameSample_new = [pathstr filesep name ext];
    imwrite(img_augmented, fileNameSample_new);
    
    %% Change Illumination
    img_augmented = augment.changeLumination(img);
    img_augmented_1 = augment.changeLumination(img);
    img_augmented_2 = augment.changeLumination(img);
    img_augmented_3 = augment.changeLumination(img);
    
    % Write on disk (same name, but with "_chgilumina0..3")
    [pathstr,name,ext] = fileparts(fileNameSample);
    name = [name '_chgilumina0'];
    fileNameSample_new = [pathstr filesep name ext];
    imwrite(img_augmented, fileNameSample_new);
    [pathstr,name,ext] = fileparts(fileNameSample);
    name = [name '_chgilumina1'];
    fileNameSample_new = [pathstr filesep name ext];
    imwrite(img_augmented_1, fileNameSample_new);
    name = [name '_chgilumina2'];
    fileNameSample_new = [pathstr filesep name ext];
    imwrite(img_augmented_2, fileNameSample_new);
    name = [name '_chgilumina3'];
    fileNameSample_new = [pathstr filesep name ext];
    imwrite(img_augmented_3, fileNameSample_new);
    
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

