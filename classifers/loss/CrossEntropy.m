classdef CrossEntropy < BaseLossFunction
    % CrossEntropy Loss implementation class        
    % https://en.wikipedia.org/wiki/Cross_entropy
    % http://stats.stackexchange.com/questions/167787/cross-entropy-cost-function-in-neural-network
    methods
        % Here the smalles value will be zero (Perfect, no loss) and the
        % biggest value is 1 (100%)
        function [lossResult] = getLoss(obj, scores, idxCorrect)
            % This implementation avoid explicit for loops (Vectorized)            
            Y_train = idxCorrect;
            sizeTraining = size(Y_train,1);
            h = scores;
            lossResult = sum(sum((-Y_train).*log(h) - (1-Y_train).*log(1-h), 2))/sizeTraining;                        
        end
    end
end

