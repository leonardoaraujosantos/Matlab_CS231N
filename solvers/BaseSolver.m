classdef (Abstract) BaseSolver < handle
    % Base class for solvers
    properties (Abstract)
        typeSolver
    end
    
    methods(Abstract, Access = public)
        [weights, timeElapsed] = optimize(obj,model);
        [type] = getType(obj);        
    end
    
end

