classdef (Sealed) NearestNeighbor < BaseClassifer
    %NEARESTNEIGHBOR Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        internal_model;
        X_train;
        Y_train;
    end
    
    properties (Access = 'private')
       sizeTrain;  
    end
    
    methods (Access = 'public')
        function obj = NearestNeighbor()
            % Nothing is learned on this classifer we just dump all
            % trainning data inside
            obj.internal_model = [];
        end
        
        function [timeElapsed] = train(obj, X_vec, Y_vec)
            % Fastest trainning possible just save all the trainning data
            obj.sizeTrain = length(X_vec);
            tic;
            obj.X_train = X_vec;
            obj.Y_train = Y_vec;
            timeElapsed = toc;
        end
        
        function [maxscore, scores, timeElapsed] = predict(obj, X_vec, distType)
            % Most slow prediction ever , it will compare distances against
            % all trainning data
            tic;            
            distanceVec = zeros(obj.sizeTrain,1);
            for idxTrain=1:obj.sizeTrain
                if (distType == 1)
                    distanceVec(idxTrain,:) = l1_distance(X_vec, obj.X_train(idxTrain,:));
                else
                    distanceVec(idxTrain,:) = l2_distance(X_vec, obj.X_train(idxTrain,:));
                end
            end
            % Get the index of the lowest distance
            [~,minIndex] = min(distanceVec);
            % Convert to Y values from the indexes
            maxscore = obj.Y_train(minIndex);
            
            % Return a list(index) of the 5 top lowest distances
            [~,fiveLess] = sort (distanceVec,1, 'ascend');
            fiveLess = fiveLess(1:5,:);
            % Convert to Y values from the indexes
            scores = obj.Y_train(fiveLess);            

            timeElapsed = toc;
        end
    end
    
end

