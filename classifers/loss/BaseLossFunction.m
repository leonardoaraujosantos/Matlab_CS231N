classdef (Abstract) BaseLossFunction < handle
    % Base class for all loss functions             
    % The Loss function will quantify how bad our current set of weights(W)
    % are, normal types of Loss Functions are SVM and Softmax.
    
    methods(Abstract, Access = public)
        % Calculate how far we are for the correct pointed by
        % classification(idxCorrect)
        [lossResult, dw] = getLoss(obj, score, correct);        
    end
    
end

