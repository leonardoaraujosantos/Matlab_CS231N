classdef (Abstract) BaseLayer < handle
    %BASELAYER Abstract class for Layer
    % Check format on reference project:
    % http://cs.stanford.edu/people/karpathy/convnetjs/
    % https://github.com/karpathy/convnetjs 
    % https://databoys.github.io/Feedforward/
    % http://scs.ryerson.ca/~aharley/neural-networks/
    
    properties (Abstract)
        typeLayer
    end
    
    methods(Abstract, Access = public)
        [result] = forward(obj);
        [gradient] = backward(obj);
        [result] = getData(obj);
        [type] = getType(obj);        
    end
    
end

