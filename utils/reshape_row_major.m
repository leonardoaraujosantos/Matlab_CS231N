function [ reshaped_matrix ] = reshape_row_major( matrix_in, shape )
%RESHAPE_ROW_MAJOR Do the reshape with row_major scan
% This is needed because other numerical libraries like numpy used
% different order (row_major) compared to matlab(col_major)
% Example:
% a = [1:1:15]
% reshape(a,[3,5])
%  1     4     7    10    13
%  2     5     8    11    14
%  3     6     9    12    15
%
% reshape_row_major(a,[3,5])
%  1     2     3     4     5
%  6     7     8     9     10
%  11    12    13    14    15

% Reshape with the shape inverted
res_trans = reshape(matrix_in,fliplr(shape));

% Transpose res_trans permuting all it's dimensions
reshaped_matrix = permute(res_trans,[ndims(res_trans):-1:1]);

end

