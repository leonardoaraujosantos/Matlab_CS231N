classdef ReluLayer < BaseLayer
    %ReluLayer Define the Relu layer
    % This layer does not have learnable parameters
        
    properties
        typeLayer 
        weights
        activations
        previousInput        
    end
    
    properties (Access = 'private')        
        numberNeurons        
        activationType
        weightsMatrix
        biasMatrix
    end
    
    methods (Access = 'public')
        function obj = ReluLayer()
            % Initialize type
            obj.typeLayer = LayerType.Relu;                        
            obj.weightsMatrix = [];                                    
        end
        
        % Get number of neurons
        function [numNeurons] = getNumNeurons(obj)
            numNeurons = obj.numberNeurons;
        end
        
        function [activations] = fp(obj,prevLayerActivations)
            activations = obj.feedForward(prevLayerActivations);
        end
        
        function [result] = feedForward(obj, input)
            obj.previousInput = input;
            result = max(0,input);
            obj.activations = result;
        end
        
        function [gradient] = backPropagate(obj, dout)            
            gradient  = dout .* (obj.previousInput >= 0);
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
            descText = sprintf('Relu\n');
        end
        
        % Get number of parameters
        function [numParameters] = getNumParameters(obj)
            numParameters = 0;
        end
    end    
end

