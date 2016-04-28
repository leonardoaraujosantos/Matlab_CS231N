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
        function [lossResult, dw] = getLoss(obj, scores, correct)
            % This implementation avoid explicit for loops (Vectorized)
            N = size(scores,1);      
            
            % Get all the correct class scores
            % Get the correct indexes
            [~, idxCorrect] = max(correct,[],2);
            correctScores = diag(scores(:,idxCorrect));
            
            % Get the margins
            margins = max(0,scores - repmat(correctScores,1,size(scores,2)) + obj.delta);
            
            % Put all the margins of the correct scores to zero
            margins(sub2ind(size(margins),[1:length(idxCorrect)]',idxCorrect)) = 0;
            
            % Loss is the sum of all elements on margin (disregarding
            % dimension) divided by the batch size
            lossResult = sum(margins(:)) / N;
            
            % Get per row the number of times that a value was positive, on
            % margin
            num_positive = sum(margins > 0);
            
            % Create a matrix dx with the same size of scores and put to
            % one on the positions where margins are bigger than zero
            dx = zeros(size(scores));
            dx(margins > 0) = 1;
            
            % Now decrement from dx the number of times that the margin was
            % positive on the position of the right score
            dx_correction = diag(dx(:,idxCorrect)) - num_positive';
            dx(sub2ind(size(dx),[1:length(idxCorrect)]',idxCorrect)) = dx_correction;
                        
            
            % dw is the derivative of the loss function over the scores
            dx = dx / N;
            dw = dx;
        end
    end
end

