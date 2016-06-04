classdef CrossEntropy < BaseLossFunction
    % CrossEntropy Loss implementation class        
    % https://en.wikipedia.org/wiki/Cross_entropy
    % http://stats.stackexchange.com/questions/167787/cross-entropy-cost-function-in-neural-network
    methods
        % Here the smalles value will be zero (Perfect, no loss) and the
        % biggest value is 1 (100%)
        function [lossResult, dx] = getLoss(obj, scores, correct)
            % This implementation avoid explicit for loops (Vectorized)                        
            N = size(correct,1);
            h = scores;
            lossResult = sum(sum((-correct).*log(h) - (1-correct).*log(1-h), 2))/N; 
            
            % dw is the derivative of the loss function over the scores
            dx = scores - correct;                        
        end
    end
end

