classdef PoolingLayer < BaseLayer
    %FULLYCONNECTEDLAYER Define the pooling layer
    % For more information refer here:
    % http://cs231n.github.io/convolutional-networks/
    
    properties
        typeLayer
        weights
        activations
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
        
        % Get number of neurons
        function [numNeurons] = getNumNeurons(obj)
            numNeurons = 0;
        end
        
        function [result] = feedForward(obj, inputs)
            result = 0;
        end
        
        function [gradient] = backPropagate(obj, targets)
            gradient = 0;
        end
        
        function [result] = getActivations(obj)
            result = obj.activations;
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
        
        % Get number of parameters (No parameters on pooling layer)
        function [numParameters] = getNumParameters(obj)
            numParameters = 0;
        end
        
        function [outSize] = getOutputSize(obj, inputSize, prevDepth)
            spatialSize = ((inputSize-obj.kernelSize)/obj.stepStride)+1;
            % The pooling layer does not change the depth of the previous
            % layer, just scale it down (Sumarize information of the
            % previous layer)
            outSize = [spatialSize spatialSize prevDepth];
        end
    end
end

