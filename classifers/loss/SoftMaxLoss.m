classdef SoftMaxLoss < BaseLossFunction
    % Softmax Loss implementation class
    
    % Softmax classifier provides "probabilities" for each class.
    % Unlike the SVM which computes uncalibrated and not easy to interpret
    % scores for all classes, the Softmax classifier allows us to compute
    % "probabilities" for all labels.    
    % https://github.com/kyunghyuncho/deepmat/blob/master/softmax.m
    methods
        % Here the smalles value will be zero (Perfect, no loss) and the
        % biggest value is 1 (100%)
        function [lossResult, dw] = getLoss(obj, scores, correct)
            % This implementation avoid explicit for loops (Vectorized)
            N = size(scores,1);
            % Improve numerical stability
            % http://cs231n.github.io/linear-classify/#softmax
            % Basically subtract the biggest value of the score (per row)
            % Hint: repmat(A,1,3) will repeat the matrix A 3 times side by
            % side, something like [A A A]
            scoresFix = scores - repmat(max(scores,[],2),1,size(scores,2));                        
            
            % Get the sum of all scores
            sumProb = sum(exp(scoresFix),2);
            sumProb = repmat(sumProb,1,size(scores,2));
            
            % Calculate probabilities
            probabilities = exp(scoresFix) ./ sumProb;            
            
            % Now we need to get the probability of every class that is
            % correct
            % http://uk.mathworks.com/matlabcentral/newsreader/view_thread/316363
            % On python(numpy) this means 
            % probs[np.arange(N), idxCorrect]
            % More info on numpy for matlab users
            % http://mathesaurus.sourceforge.net/matlab-numpy.html
            % Get the correct indexes
            [~, idxCorrect] = max(correct);
            % Select a particular element from each row of a matrix based 
            % on another matrix 
            probCorrect = diag(probabilities(:,idxCorrect));
            
            % Now calculate the loss
            lossResult = -sum(log(probCorrect))/N;
            
            % dw is the derivative of the loss function over the scores
            dw = probCorrect;
            dw = dw - 1;
            dw = dw/N;
        end
    end
end

