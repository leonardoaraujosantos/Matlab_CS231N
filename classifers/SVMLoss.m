classdef SVMLoss < BaseLossFunction
    % SVM Loss implementation class
    % The Multiclass Support Vector Machine "wants" the score of the
    % correct class to be higher than all other scores by at least a
    % margin of delta. If any class has a score inside the red region
    % (or higher), then there will be accumulated loss.
    % Otherwise the loss will be zero. Our objective will be to find the
    % weights that will simultaneously satisfy this constraint for all
    % examples in the training data and give a total loss that is as
    % low as possible.
    properties (Access = 'private')
        delta;
    end
    
    methods
        function obj = SVMLoss(parDelta)
            % Initialize SVM loss hyper-parameter (Bad one more thing to
            % think...)
            obj.delta = parDelta;
        end
        
        % Here the smalles value will be zero (Perfect, no loss) and the
        % biggest value is unbounded
        function [lossResult] = getLoss(obj, scores, idxCorrect)
            % Get the correct score
            correct_score = scores(idxCorrect);
            
            % Get all scores except the wrong one
            % http://stackoverflow.com/questions/19596268/select-all-elements-except-one-in-a-vector-matlab
            incorrect_scores = scores(1:end ~= idxCorrect);
            lossResult = sum(max(0,(incorrect_scores - correct_score) + obj.delta));
        end
    end
end

