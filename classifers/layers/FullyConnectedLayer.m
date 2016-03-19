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
        activationType
        weightsMatrix
        biasMatrix
    end
    
    methods (Access = 'public')
        function obj = FullyConnectedLayer(pNumNeurons, pActType)
            % Initialize type
            obj.typeLayer = LayerType.FullyConnected;
            obj.numberNeurons = pNumNeurons;
            obj.activationType = pActType;
            
            % We have one bias unit per neuron
            obj.biasMatrix = single(ones(pNumNeurons,1));
            
            % The number of weights per/layer depends on the number of
            % neurons on the previous layer (NumNeuronsL1 * NumNeuronsL2)
            % But the number of weights per/neron on the layer is exactly
            % the number of neurons on the previous layer
            obj.weightsMatrix = [];
            
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
        
        function [descText] = getDescription(obj)
            [~, names] = enumeration('ActivationType');
            descText = sprintf('FC num_neurons=%d Activation=%s\n',obj.numberNeurons,names{obj.activationType});
        end
    end    
end

