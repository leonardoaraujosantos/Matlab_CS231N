function [ resp ] = l1_distance( I1, I2 )
%L1_DISTANCE Return the Manhatan distance between the vectors I1 and I2

% Vectorized version
resp = sum(abs(I1-I2));

end

