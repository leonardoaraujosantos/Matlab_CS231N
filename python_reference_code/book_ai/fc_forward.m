function [out, cache] = fc_forward(x,w,b)
cache = {x,w,b};
% Get the batchsize
lenSizeActivations = length(size(x));
szX = size(x);
if (lenSizeActivations < 3)
    N = szX(1);
else
    N = size(x,ndims(x));
end

% Reshape activations to [Nx(d_1, ..., d_k)], which will be a 2d matrix
% [NxD]
D = prod(szX(1:end-1));
res_x = reshape_row_major(x,[N, D]);

out = (res_x*w) + (repmat(b,size(res_x,1),1));
end
