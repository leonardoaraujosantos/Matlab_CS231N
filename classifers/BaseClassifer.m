classdef (Abstract) BaseClassifer < handle
    % Base class for all classifiers            
    
    methods(Abstract, Access = public)
        [maxscore, scores, timeElapsed] = predict(obj);
        
        % Video tutorials about gradient-descent, mini-batch
        % https://www.youtube.com/watch?v=GvHmwBc9N30
        % https://class.coursera.org/ml-003/lecture/104
        % https://class.coursera.org/ml-003/lecture/105
        % https://class.coursera.org/ml-003/lecture/106
        % Andrew Ng quote....
        % It's not about having the better algorithm, is who has more data
        [timeElapsed] = train(obj);
    end
    
end

