function startup()
disp('Preparing project');

%% Add folders on the matlab path
curdir = fileparts(mfilename('fullpath'));
addpath(genpath(fullfile(curdir, 'utils')));
addpath(genpath(fullfile(curdir, 'tests')));
addpath(genpath(fullfile(curdir, 'math_utils')));
addpath(genpath(fullfile(curdir, 'assignements')));
addpath(genpath(fullfile(curdir, 'classifers')));

end