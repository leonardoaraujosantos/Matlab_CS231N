classdef DeepLearningModel < handle
    %DEEPLEARNINGMODEL Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        layers
    end
    
    methods
        function [obj] = DeepLearningModel(layers)
           obj.layers = layers; 
        end
        
        function [maxscore, scores, timeElapsed] = loss(obj, X_vec)
            maxscore = 0;
            scores = 0;
            timeElapsed = 0;
        end
                
    end
    
end

