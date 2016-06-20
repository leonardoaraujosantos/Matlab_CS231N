function [ filenames ] = getFilenamesFromDirectory( input_args )
% Get all filenames from a particular directory
filenames = {};
if (isunix) %# Linux, mac
    [~, result] = system( ['find ', input_args, ' -type f'] );
    result = strsplit(result);
    filenames = result;        
elseif (ispc) %# Windows
    % TODO
    % type c:\windows\win.ini | find /c /v "~~~"
end

end

