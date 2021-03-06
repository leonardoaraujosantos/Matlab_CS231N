classdef OutputLayer < BaseLayer
    %FULLYCONNECTEDLAYER Define the output softmax layer        
    properties
        typeLayer
        weights
        activations
        z
    end
    
    properties (Access = 'private')        
        numClasses
        activationObject
        activationType
    end
        
    methods (Access = 'public')
        function obj = OutputLayer(pNumClasses, pActType)
            % Initialize type
            obj.typeLayer = LayerType.Output;
            obj.numClasses = pNumClasses;  
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
        
        % Get number of neurons
        function [numNeurons] = getNumNeurons(obj)
            numNeurons = obj.numClasses;
        end
        
        function [result] = feedForward(obj, activations, theta)
            % theta are the weights of the previous layer
            % Include the bias (1)
            sizeRows = size(activations, 1);            
            activations = [ones(sizeRows, 1) activations];            
            % The multiplication gives the same result of the dot product
            % but faster (=~ 2x)
            obj.z = activations * theta';
            result = obj.activationObject.forward_prop(obj.z);           
            obj.activations = result;
        end
        
        function [gradient] = backPropagate(obj)
            gradient = obj.activationObject.back_prop([ones(size(obj.z, 1), 1) obj.z]);            
        end
        
        % This will return the scores
        function [result] = getActivations(obj)
            result = obj.activations;
        end                
        
        function [type] = getType(obj)
            type = obj.typeLayer;
        end
        
        function [loss] = getLossFunction(obj)
            loss = obj.lossFunction;
        end
        
        % Get number of parameters (No parameters on pooling layer)
        function [numParameters] = getNumParameters(obj)
            numParameters = numel(obj.weights);
        end
        
        function [descText] = getDescription(obj)
            [~, names] = enumeration('ActivationType');
            descText = sprintf('OUTPUT num_classes=%d Activation=%s\n',obj.numClasses,names{obj.activationType});
        end
    end    
end

