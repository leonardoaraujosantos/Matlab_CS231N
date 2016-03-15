classdef (Abstract) BaseLayer < handle
    %BASELAYER Abstract class for Layer
    % Check format on reference project:
    % http://cs.stanford.edu/people/karpathy/convnetjs/
    % https://github.com/karpathy/convnetjs    
    
    methods(Abstract, Access = public)
        [result] = forward();
        [gradient] = backward();
        getData();
        getType();
        setType();
    end
    
end

