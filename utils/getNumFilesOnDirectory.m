function [ numlines ] = getNumFilesOnDirectory( input_directory )
%GETNUMFILESONDIRECTORY Get the number of lines on the file, but using
% Operation system commands. The idea is to be faster
% Ex:
% numFiles = getNumFilesOnFile('/mnt/fs3/QA_analitics/For_Leo/Train.txt')

numlines = 0;
if (isunix) %# Linux, mac
    [~, result] = system( ['find ', input_directory, ' -type f | wc -l '] );
    result = strsplit(result);
    numlines = str2num(result{1});
    
elseif (ispc) %# Windows
    % TODO
    % type c:\windows\win.ini | find /c /v "~~~"
    numlines = 0;    
end


end

