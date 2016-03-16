classdef (Abstract) BaseActivationFunction < handle
    % Base class for all activation functions                
    methods(Abstract, Access = public, Static)
        [result] = forward_prop(x);
        [result] = back_prop(x);
    end
    
end



