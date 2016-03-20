classdef ConvolutionalLayer < BaseLayer
    %FULLYCONNECTEDLAYER Define the convolutional layer
    %   Actually this is the normal hidden layer on Neural Networks
    % More information:
    % http://www.slideshare.net/deview/251-implementing-deep-learning-using-cu-dnn
    
    properties
        typeLayer
        weights
        activations
    end
    
    properties (Access = 'private')
        activationObject
        kernelSize
        numFilters
        stepStride
        numPad
        activationType
    end
    
    methods (Access = 'public')
        function obj = ConvolutionalLayer(pKernelSize, pFilters, pStride, pPad, pActType)
            % Initialize type
            obj.typeLayer = LayerType.Convolutional;
            obj.kernelSize = pKernelSize;
            obj.numFilters = pFilters;
            obj.stepStride = pStride;
            obj.numPad = pPad;
            obj.activationType = pActType;
            
            switch pActType
                case ActivationType.Sigmoid
                    obj.activationObject = SigmoidActivation();
                case ActivationType.Tanh
                    obj.activationObject = TanhActivation();
                case ActivationType.Relu
                    obj.activationObject = ReluActivation();
                otherwise
                    obj.activationObject = SigmoidActivation();
            end
        end
        
        function [result] = feedForward(obj, inputs)
            result = 0;
        end
        
        function [gradient] = backPropagate(obj, targets)
            gradient = 0;
        end
        
        function [result] = getActivations(obj)
            result = 0;
        end
        
        function [type] = getType(obj)
            type = obj.typeLayer;
        end
        
        function [numN] = getKernelSize(obj)
            numN = obj.kernelSize;
        end
        
        function [numN] = getNumberOfFilters(obj)
            numN = obj.numFilters;
        end
        
        function [numN] = getStride(obj)
            numN = obj.stepStride;
        end
        
        function [numN] = getPad(obj)
            numN = obj.numPad;
        end
        
        function [actFunc] = getActivation(obj)
            actFunc = obj.activationObject;
        end
        
        % Get number of neurons
        function [numNeurons] = getNumNeurons(obj)
            numNeurons = [];
        end
         
        function [descText] = getDescription(obj)
            [~, names] = enumeration('ActivationType');            
            descText = sprintf('CONV ksize=%d num_filters=%d stride=%d num_pad=%d Activation=%s\n',...
                obj.kernelSize,obj.numFilters,obj.stepStride,...
                obj.numPad,names{obj.activationType});
        end
    end
end

