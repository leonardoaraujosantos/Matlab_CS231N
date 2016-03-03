function startup()
disp('Preparing project');
curdir = fileparts(mfilename('fullpath'));
addpath(genpath(fullfile(curdir, 'utils')));
addpath(genpath(fullfile(curdir, 'tests')));
addpath(genpath(fullfile(curdir, 'math_utils')));

end