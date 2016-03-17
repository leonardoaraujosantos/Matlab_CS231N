classdef (Sealed) DeepNeuralNetwork < BaseClassifer
    %DEEPNEURALNETWORK Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        layers
    end
    
    methods
        function obj = DeepNeuralNetwork(layers)
            obj.layers = layers;
        end
        
        function [timeElapsed] = train(obj, X_vec, Y_vec)            
            tic;
            1+1;
            timeElapsed = toc;
        end
        
        function [maxscore, scores, timeElapsed] = predict(obj, X_vec)            
            tic;
            maxscore = 0;
            scores = [];
            timeElapsed = toc;
        end
    end
    
end

