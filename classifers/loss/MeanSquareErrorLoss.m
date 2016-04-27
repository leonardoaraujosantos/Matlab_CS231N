classdef MeanSquareErrorLoss < BaseLossFunction
    % Square Error Loss implementation class    
        
    methods                
        % Here the smalles value will be zero (Perfect, no loss) and the
        % biggest value is unbounded
        function [lossResult, dw] = getLoss(obj, scores, correct)
            N = size(scores,1);                        
            lossResult = sum((scores - correct).^2);
            lossResult = lossResult / N;
            
            % Derivative of loss related to the scores
            dw = scores - correct;
            dw = dw/N;
        end
    end
end

