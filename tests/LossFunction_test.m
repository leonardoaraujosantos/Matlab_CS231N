%% Loss Function
% The loss function quantifies our unhappiness with predictions on the 
% training set
% A loss function or cost function is a function that maps an event or 
% values of one or more variables onto a real number intuitively 
% representing some "cost" associated with the event. 
% An optimization problem seeks to minimize a loss function.
% 
% <<../../docs/imgs/LossFuncPlot.png>>
%
%
% $L =  { \frac{1}{N} \sum_i L_i } + { \lambda R(W) }\\$
%
% More info at:
% http://cs231n.github.io/linear-classify/#loss

%% Test 1: Square error loss
% $L_i = \sum_{j\neq y_i} \max(0, s_j - s_{y_i} + \Delta)$
% 
% <<../../docs/imgs/SVM_vs_Softmax.png>>
%
scores = [-2.85, 0.86, 0.28];
correct = [0 0 1];

% Initialize object suqaredErrorLoss
testLossFunction = SquareErrorLoss();

% Calculate loss for correct class
result = testLossFunction.getLoss(scores,correct);
fprintf('Result(Suared Error) is %d\n',result);


%% Test 2: Test Softmax loss
% $L_i = -\log(\frac{e^{f_{y_i}}}{ \sum_j e^{f_j} })$
% 
% <<../../docs/imgs/SVM_vs_Softmax.png>>
%
% For more information on publish tags, refer to
% http://uk.mathworks.com/help/matlab/matlab_prog/marking-up-matlab-comments-for-publishing.html
% Also for generating latex you can use this command:
% http://uk.mathworks.com/help/symbolic/latex.html
scores = [-2.85, 0.86, 0.28; 0 0.3 0.2; 0.3 -0.3 0];
correct = [0 0 1; 0 1 0; 1 0 0];
% Correct class should be 3
idx_correct = 3;
% Correct response, check on SVM vs Softmax paragraph
% http://cs231n.github.io/linear-classify/#softmax
ref_result = 1.04; 

% Initialize object SoftMax
testLossFunction = SoftMaxLoss();

% Calculate loss for correct class
result = testLossFunction.getLoss(scores,correct);
fprintf('Result(SoftMax) is %d correct is %d\n',result, ref_result);

error = abs(ref_result - result);
fprintf('Error is %d\n',error);
assert (error < 0.001);

%% Test 3: Test Svm loss
% $L_i = \sum_{j\neq y_i} \max(0, s_j - s_{y_i} + \Delta)$
% 
% <<../../docs/imgs/SVM_vs_Softmax.png>>
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
