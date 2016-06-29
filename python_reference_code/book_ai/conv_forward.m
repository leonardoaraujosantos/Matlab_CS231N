function [out, cache] = conv_forward(x,w,b,params)

% Cache parameters and inputs for backpropagation
cache = {x,w,b,params};

% N (input volume), F(output volume)
% C channels
% H (rows), W(cols)
[H, W, C, N] = size(x);
[HH, WW, C, F] = size(w);

% Calculate output size, and allocate result
H_R = ((H + (2*params.numPad) - HH) / params.stepStride) + 1;
W_R = ((W + (2*params.numPad) - WW) / params.stepStride) + 1;
out = zeros(H_R,W_R,F,N);

% Pad if needed
if (params.numPad > 0)
    x_pad = padarray(x,[params.numPad params.numPad 0 0]);
end

% Convolve for each input/output depth
for idxBatch=1:N
    for idxFilter=1:F
        
        % Select weights and inputs
        weights = w(:,:,:,idxFilter);
        input = x_pad(:,:,:,idxBatch);
        
        % Do naive(slow) convolution)
        resConv = convn_vanilla(input,weights,params.stepStride);
        
        % Add bias and store
        out(:,:,idxFilter,idxBatch) = resConv + b(idxFilter);
    end
end

end
