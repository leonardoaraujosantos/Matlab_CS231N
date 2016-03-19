classdef (Abstract) BaseLayerContainer < handle
    %BASELAYER Abstract class for Layer container
    % Check format on reference project:
    % http://cs.stanford.edu/people/karpathy/convnetjs/
    % https://github.com/karpathy/convnetjs            
    
    methods(Abstract, Access = public)
        pushLayer(obj,metaDataLayer);
        removeLayer(obj,index);
        getData(obj,index);
        showStructure(obj);
    end
    
end

