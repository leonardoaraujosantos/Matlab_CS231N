classdef SquareErrorLoss < BaseLossFunction
    % Square Error Loss implementation class    
        
    methods                
        % Here the smalles value will be zero (Perfect, no loss) and the
        % biggest value is unbounded
        function [lossResult] = getLoss(obj, scores, idxCorrect)
            % Get the correct score
            correct_score = scores(idxCorrect);
                        
            lossResult = sum((correct_score - incorrect_scores)^2);
        end
    end
end

