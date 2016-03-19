classdef PoolingLayer < BaseLayer
    %FULLYCONNECTEDLAYER Define the pooling layer
    % For more information refer here:
    % http://cs231n.github.io/convolutional-networks/
    
    properties
        typeLayer
    end
    
    properties (Access = 'private')
        kernelSize
        stepStride
        poolingType
    end
    
    methods (Access = 'public')
        function obj = PoolingLayer(pKernelSize, pStride, pPoolType)
            % Initialize type
            obj.typeLayer = LayerType.Pooling;
            obj.kernelSize = pKernelSize;
            obj.stepStride = pStride;
            obj.poolingType = pPoolType;                        
        end
        
        function [result] = feedForward(obj, inputs)
            result = 0;
        end
        
        function [gradient] = backPropagate(obj, targets)
            gradient = 0;
        end
        
        function [result] = getData(obj)
            result = 0;
        end
        
        function [type] = getType(obj)
            type = obj.typeLayer;
        end
        
        function [numN] = getKernelSize(obj)
            numN = obj.kernelSize;
        end
        
        function [numN] = getStride(obj)
            numN = obj.stepStride;
        end
        
        function [numN] = getPoolingType(obj)
            numN = obj.poolingType;
        end
        
        function [descText] = getDescription(obj)
            [~, names] = enumeration('PoolingType');            
            descText = sprintf('POOL ksize=%d stride=%d Type=%s\n',...
                obj.kernelSize,...
                obj.stepStride,names{obj.poolingType});
        end
    end
end

