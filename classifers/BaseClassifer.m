classdef (Abstract) BaseClassifer < handle
    % Base class for all classifiers    
    
    properties (Abstract)
        internal_model
    end
    
    methods(Abstract, Access = public)
        [maxscore, scores, timeElapsed] = predict;
        [timeElapsed] = train;
    end
    
end

