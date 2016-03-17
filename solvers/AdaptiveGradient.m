classdef AdaptiveGradient < BaseSolver
    % Implementation of ADAGRAD
    
    properties
        typeSolver
    end
    
    methods(Abstract, Access = public)
        function [weights, timeElapsed] = optimize(obj,model)
            tic;
            timeElapsed = toc;
        end
        [type] = getType(obj);        
    end
end

