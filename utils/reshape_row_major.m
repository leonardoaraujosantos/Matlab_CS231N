function [ reshaped_matrix ] = reshape_row_major( matrix_in, shape )
%RESHAPE_ROW_MAJOR Do the reshape with row_major scan
% This is needed because other numerical libraries like numpy used
% different order (row_major) compared to matlab(col_major)
res_trans = reshape(matrix_in,fliplr(shape));
reshaped_matrix = permute(res_trans,[ndims(res_trans):-1:1]);

end

