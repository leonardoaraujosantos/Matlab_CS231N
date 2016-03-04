function [ tb_batch_cifar ] = loadCifar_10( batch_filename, meta_filename )
% Create a table from the cifar batch m file

% Load mat files
load(batch_filename);
load(meta_filename);

% Get number of elements on data and allocate vectors
numElements = length(data);
vecImgs = cell(numElements,1);
vecIds = zeros(numElements,1);
vecImgsData = cell(numElements,1);
vecClassDesc = cell(numElements,1);

% Iterate on the data vector
for idx=1:numElements
    % Extract information from data
    img_vec = data(idx,:);
    img = reshape(data(idx,:), [32 32 3]);
    id_class = labels(idx);
    class_desc = label_names{id_class+1};
    % The image was saved on row-major order, so we need to rotate-it
    vecImgs{idx,:} = imrotate(img,-90);
    vecIds(idx,:) = id_class;
    vecImgsData{idx,:} = img_vec;
    vecClassDesc{idx,:} = class_desc;
end
tb_batch_cifar = table;
tb_batch_cifar.Y = vecIds;
tb_batch_cifar.Desc = vecClassDesc;
tb_batch_cifar.Image = vecImgs;
tb_batch_cifar.X = vecImgsData;

end

