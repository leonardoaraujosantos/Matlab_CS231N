%% Test 1: Test SoftMax
% 
% <</home/leo/work/Matlab_CS231N/docs/imgs/SVM_vs_Softmax.png>>
%
% For more information on publish tags, refer to
% http://uk.mathworks.com/help/matlab/matlab_prog/marking-up-matlab-comments-for-publishing.html
scores = [-2.85, 0.86, 0.28];
% Correct class should be 3
idx_correct = 3;
% Correct response, check on SVM vs Softmax paragraph
% http://cs231n.github.io/linear-classify/#softmax
ref_result = 1.04; 

% Initialize object SoftMax
testLossFunction = SoftMaxLoss();

% Calculate loss for correct class
result = testLossFunction.getLoss(scores,idx_correct);
fprintf('Result(SoftMax) is %d correct is %d\n',result, ref_result);

error = abs(ref_result - result);
fprintf('Error is %d\n',error);
assert (error < 0.001);

%% Test 2: Test SVM (hinge loss)
% 
% <</home/leo/work/Matlab_CS231N/docs/imgs/SVM_vs_Softmax.png>>
%
scores = [-2.85, 0.86, 0.28];
% Correct class should be 3
idx_correct = 3;
% Correct response, check on SVM vs Softmax paragraph
% http://cs231n.github.io/linear-classify/#softmax
ref_result = 1.58; 

% Initialize object SoftMax
testLossFunction = SVMLoss(1);

% Calculate loss for correct class
result = testLossFunction.getLoss(scores,idx_correct);
fprintf('Result(SVM) is %d correct is %d\n',result, ref_result);

error = abs(ref_result - result);
fprintf('Error is %d\n',error);
assert (error < 0.001);