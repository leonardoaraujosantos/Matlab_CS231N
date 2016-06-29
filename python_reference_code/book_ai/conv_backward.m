function [ dx, dw, db ] = conv_backward( dout, cache )
x = cache{1}; w = cache{2}; b = cache{3}; params = cache{4};
% N (Batch size), F(output volume, number of filters)
% C channels (input volume)
% H (rows), W(cols)
[H_R, W_R,F, N] = size(dout);
[H, W,C, N] = size(x);
[HH, WW,C, F] = size(w);
S = params.stepStride;
% Pad if needed
if (params.numPad > 0)
    x = padarray(x,[params.numPad params.numPad 0 0]);
end

dx = zeros(size(x)); dw = zeros(size(w)); db = zeros(size(b));

% Calculate dx
for n=1:N
    for depth=1:F
        weights = w(:,:,:,depth);
        for r=1:S:H
            for c=1:S:W
                input = dout(ceil(r/S),ceil(c/S),depth,n);
                prod =  weights * input;
                dx(r:(r+HH)-1,c:(c+WW)-1,:,n) = dx(r:(r+HH)-1,c:(c+WW)-1,:,n) + prod;
            end
        end
    end
end
% Delete padded rows
dx = dx(1+params.numPad:end-params.numPad, 1+params.numPad:end-params.numPad,:,:);

% Calculate dw
for n=1:N
    for depth=1:F
        for r=1:H_R
            for c=1:W_R
                input = dout(r,c,depth,n);
                weights = x(r*S:(r*S+HH)-1,c*S:(c*S+WW)-1,:,n);
                prod =  weights * input;
                dw(:,:,:,depth) = dw(:,:,:,depth) + prod;
            end
        end
    end
end

% Calculate db
for depth=1:F
    selDoutDepth = dout(: , : , depth, :);
    db(depth) = sum( selDoutDepth(:) );
end

end

