classdef OutputRegressionLayer < BaseLayer
    %FULLYCONNECTEDLAYER Define the output regression layer        
    % Regression layers does not output a binary class but a number(range)
    properties
        typeLayer      
        weights
        activations
    end
    
    properties (Access = 'private')        
        numberNeurons                  
    end
        
    methods (Access = 'public')
        function obj = OutputRegressionLayer(pNumNeurons)
            % Initialize type
            obj.typeLayer = LayerType.OutputRegression;
            obj.numberNeurons = pNumNeurons;                
        end
        
        % Get number of neurons
        function [numNeurons] = getNumNeurons(obj)
            numNeurons = obj.numberNeurons;
        end
        
        function [result] = feedForward(obj, inputs)
            result = [];
        end        
        
        function [gradient] = backPropagate(obj, targets)
            gradient = [];
        end
        
        % This will return the scores
        function [result] = getActivations(obj)
            result = obj.activations;
        end                
        
        function [type] = getType(obj)
            type = obj.typeLayer;
        end
        
        function [numN] = getNumberofNeurons(obj)
            numN = obj.numberNeurons;
        end
        
        function [descText] = getDescription(obj)            
            descText = sprintf('REGRESSION num_neurons=%d Activation=%s\n',obj.numberNeurons);
        end
        
        % Get number of parameters
        function [numParameters] = getNumParameters(obj)
            numParameters = numel(obj.weights);
        end
    end    
end

