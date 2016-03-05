function [ resp ] = l1_distance( I1, I2 )
%L1_DISTANCE Return the Manhatan distance between the vectors I1 and I2
% L1 distance is better than L2 distance for large arrays

% Vectorized version
resp = sum(abs(I1-I2));

end

