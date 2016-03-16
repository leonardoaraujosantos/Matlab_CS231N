classdef (Abstract) BaseClassifer < handle
    % Base class for all classifiers    
    
    properties (Abstract)
        internal_model
    end
    
    methods(Abstract, Access = public)
        [maxscore, scores, timeElapsed] = predict;
        
        % Video tutorials about gradient-descent, mini-batch
        % https://www.youtube.com/watch?v=GvHmwBc9N30
        % https://class.coursera.org/ml-003/lecture/104
        % https://class.coursera.org/ml-003/lecture/105
        % https://class.coursera.org/ml-003/lecture/106
        % Andrew Ng quote....
        % It's not about having the better algorithm, is who has more data
        [timeElapsed] = train;
    end
    
end

