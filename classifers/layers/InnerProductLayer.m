classdef InnerProductLayer < BaseLayer
    %InnerProductLayer Define the fully connected layer    
    
    properties
        typeLayer 
        weights
        activations
        z
        dropoutMask
        previousInput
    end
    
    properties (Access = 'private')        
        numberNeurons
        activationObject
        activationType
        weightsMatrix
        biasMatrix
    end
    
    methods (Access = 'public')
        function obj = InnerProductLayer()
            % Initialize type
            obj.typeLayer = LayerType.FullyConnected;            
        end
        
        % Get number of neurons
        function [numNeurons] = getNumNeurons(obj)
            numNeurons = obj.numberNeurons;
        end
        
        function [result] = feedForward(obj, activations, theta, biasWeights)
            % Matlab reshape order is not the same as numpy, so to make the
            % same you need to transpose the dimensions of the input, than
            % reshape on the reverse order then transpose...
            % TODO: This was done to make results compatible with cs231n
            % assigments but in the end of the day is just a dot product
            % between activations and theta, plus 1*biasWeights, I dont
            % think that the order will change the results ...
            % a1 = reshape(activations,size(activations,1),[])
            % permuting all dimensions [4 3 2 1] 
            a1 = reshape(permute(activations,[ndims(activations):-1:1]),[],size(activations,1))';
            result = (a1*theta) + (repmat(biasWeights,size(a1,1),1));
            
            % Save the previous inputs for use later on backpropagation
            obj.previousInput = activations;
            
        end
        
        function [gradient] = backPropagate(obj)
            gradient = obj.activationObject.back_prop([ones(size(obj.z, 1), 1) obj.z]);
        end
        
        function [result] = getActivations(obj)
            result = obj.activations;
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
        
        % Get number of parameters
        function [numParameters] = getNumParameters(obj)
            numParameters = numel(obj.weights);
        end
    end    
end

