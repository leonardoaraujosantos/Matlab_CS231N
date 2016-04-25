function startup()
disp('Preparing project');

%% Add folders on the matlab path
curdir = fileparts(mfilename('fullpath'));
addpath(genpath(fullfile(curdir, 'utils')));
addpath(genpath(fullfile(curdir, 'datasets')));
addpath(genpath(fullfile(curdir, 'tests')));
addpath(genpath(fullfile(curdir, 'math_utils')));
addpath(genpath(fullfile(curdir, 'assignements')));
addpath(genpath(fullfile(curdir, 'classifers')));
addpath(genpath(fullfile(curdir, 'solvers')));
addpath(genpath(fullfile(curdir, 'simple_examples')));

% Add python folder
addpath(genpath(fullfile(curdir, 'python_reference_code')));
insert(py.sys.path,int32(0),[pwd filesep 'python_reference_code']);

end