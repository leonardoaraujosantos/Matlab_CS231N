function [ resp ] = l2_distance( I1, I2 )
%L2_DISTANCE Return the euclidian distance between the vectors I1 and I2

% Vectorized version
resp = sum((I1-I2).^2).^0.5;
% resp = norm(I1-I2);
% resp = sqrt(sum((I1-I2).^2));


end

