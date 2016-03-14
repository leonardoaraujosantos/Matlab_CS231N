classdef SoftMaxLoss < BaseLossFunction
    % Softmax Loss implementation class
    
    % Softmax classifier provides "probabilities" for each class.
    % Unlike the SVM which computes uncalibrated and not easy to interpret 
    % scores for all classes, the Softmax classifier allows us to compute 
    % "probabilities" for all labels.
    
    methods
        % Here the smalles value will be zero (Perfect, no loss) and the
        % biggest value is 1 (100%)
        function [lossResult] = getLoss(obj, scores, idxCorrect)
            % This implementation avoid explicit for loops (Vectorized)
            
            % Improve numerical stability 
            % http://cs231n.github.io/linear-classify/#softmax
            score_new = scores - max(scores);
            
            probabilities = exp(score_new)/sum(exp(score_new));
            
            lossResult = -log(probabilities(idxCorrect));
        end
    end
    
end

