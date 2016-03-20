classdef InputLayer < BaseLayer
    %FULLYCONNECTEDLAYER Define the fully connected layer
    %   Actually this is the normal hidden layer on Neural Networks
    % More information:
    % http://www.slideshare.net/deview/251-implementing-deep-learning-using-cu-dnn
    
    properties 
        typeLayer
        weights
        activations
    end
    
    properties (Access = 'private')        
        numRows
        numCols
        depthInput        
    end
    
    methods (Access = 'public')
        function obj = InputLayer(pRows, pCols, pDepth)
            % Initialize type
            obj.typeLayer = LayerType.Input;
            obj.numRows = pRows;
            obj.numCols = pCols;
            obj.depthInput = pDepth;            
            % The activations on the input layer are actually the data
            obj.activations = [];
        end
        
        % Get number of neurons
        function [numNeurons] = getNumNeurons(obj)
            numNeurons = obj.numRows * obj.numCols * obj.depthInput;
        end
        
        % There is no need to forward the input layer (Use getData)
        function [result] = feedForward(obj, inputs)
            result = [];
        end
        
        % There is no need to backpropagate the input layer
        function [gradient] = backPropagate(obj, targets)
            gradient = [];
        end
        
        % Get the current data set on the constructor
        function [result] = getActivations(obj)
            result = obj.activations;
        end
        
        % Get the current data set on the constructor
        function setActivations(obj, pData)  
            % Input X is a row vector
            obj.activations = pData';
        end
        
        function [type] = getType(obj)
            type = obj.typeLayer;
        end
        
        function [descText] = getDescription(obj)
            descText = sprintf('INPUT %dx%d depth=%d\n',obj.numRows, obj.numCols, obj.depthInput);
        end
    end    
end

