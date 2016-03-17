classdef OutputRegressionLayer < BaseLayer
    %FULLYCONNECTEDLAYER Define the output regression layer        
    % Regression layers does not output a binary class but a number(range)
    properties
        typeLayer        
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
        
        function [result] = feedForward(obj, inputs)
            result = [];
        end        
        
        function [gradient] = backPropagate(obj, targets)
            gradient = [];
        end
        
        % This will return the scores
        function [result] = getData(obj)
            result = [];
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
    end    
end

