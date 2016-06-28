function [ dx, dw, db ] = fc_backward( dout, cache )
x = cache{1}; w = cache{2}; b = cache{3};

% Get the batchsize
lenSizeActivations = length(size(x));
szX = size(x);
if (lenSizeActivations < 3)
    N = szX(1);
else
    N = size(x,ndims(x));
end

% Get dX (Same format as x)
dx = dout * w';
szX = size(x);
dx = reshape_row_major(dx,szX);

% Get dW (Same format as w)
% Reshape activations to [Nx(d_1, ..., d_k)], which will be a 2d matrix
% [NxD]
szX = size(x);
D = prod(szX(1:end-1));
res_x = reshape_row_major(x,[N, D]);
dw = res_x' * dout;

% Get dB (Same format as x)
% Sum all columns of dout
db = sum(dout,1);

end

