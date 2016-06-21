function [ filenames ] = getFilenamesFromDirectory( input_args )
% Get all filenames from a particular directory
filenames = {};
result = dir(input_args);
szResult = numel(result);
countFiles = 1;
for idxRes=1:szResult
   if result(idxRes).isdir == 0
       filenames{countFiles} = [input_args, '/', result(idxRes).name];
       countFiles = countFiles + 1;
   end
end



end

