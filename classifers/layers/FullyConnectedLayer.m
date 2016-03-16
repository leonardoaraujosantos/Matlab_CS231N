classdef FullyConnectedLayer < BaseLayer
    %FULLYCONNECTEDLAYER Define the fully connected layer
    %   Actually this is the normal hidden layer on Neural Networks
    % More information:
    % http://www.slideshare.net/deview/251-implementing-deep-learning-using-cu-dnn
    
    properties
        typeLayer        
    end
    
    properties (Access = 'private')        
        numberNeurons
        activationObject
    end
    
    methods (Access = 'public')
        function obj = FullyConnectedLayer(pNumNeurons, pActType)
            % Initialize type
            obj.typeLayer = LayerType.FullyConnected;
            obj.numberNeurons = pNumNeurons;
            
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
        
        function [result] = forward(obj)
            result = 0;
        end
        
        function [gradient] = backward(obj)
            gradient = 0;
        end
        
        function [result] = getData(obj)
            result = 0;
        end
        
        function [type] = getType(obj)
            type = obj.typeLayer;
        end
        
        function [numN] = getNumberofNeurons(obj)
            numN = obj.numberNeurons;
        end
        
        function [actFunc] = getActivation(obj)
            actFunc = obj.activationObject;
        end
    end
    
end

