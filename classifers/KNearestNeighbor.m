classdef (Sealed) KNearestNeighbor < BaseClassifer
    % Implement K-Nearest Neighbor
    %   Detailed explanation goes here
    
    properties
        internal_model;
        X_train;
        Y_train;
    end
    
    properties (Access = 'private')
        sizeTrain;
    end
    
    methods (Access = 'private')
        function res = strVoting(obj,inputData)
            % Implement voting system on case of cell of strings
            numElements = size(inputData,1);
            voteArray = zeros(numElements,1);
            for idxEls = 1:numElements
                element = inputData{idxEls,:};
                countTimes = 0;
                % Look for element on original array
                for idxLookEls=1:numElements
                    elementLook = inputData{idxLookEls,:};
                    if strcmp(elementLook,element) == 1
                        countTimes = countTimes + 1;
                    end
                end
                voteArray(idxEls) = countTimes;
            end
            [~,kMostVote] = sort (voteArray,1, 'descend');
            kMostVote = kMostVote(1,:);
            res = inputData(kMostVote,:);           
        end
    end
        
        methods (Access = 'public')
            function obj = KNearestNeighbor()
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
            
            function [maxscore, scores, timeElapsed] = predict(obj, X_vec, distType, K)
                % Most slow prediction ever , it will compare distances against
                % all trainning data
                tic;
                scores = [];
                distanceVec = zeros(obj.sizeTrain,1);
                for idxTrain=1:obj.sizeTrain
                    if (distType == 1)
                        distanceVec(idxTrain,:) = l1_distance(X_vec, obj.X_train(idxTrain,:));
                    else
                        distanceVec(idxTrain,:) = l2_distance(X_vec, obj.X_train(idxTrain,:));
                    end
                end
                
                % Instead of finding the single closest image in the training
                % set, we will find the top k closest images, and have them
                % vote on the label of the test image.
                % Return a list(index) of the K top lowest distances
                [~,kLessDistantIdx] = sort (distanceVec,1, 'ascend');
                kLessDistantIdx = kLessDistantIdx(1:K,:);
                
                % Convert to Y values from the indexes
                kLessDistant = obj.Y_train(kLessDistantIdx);
                if K == 1
                    maxscore = kLessDistant;
                else
                    % Get the most occurent value (most voted)
                    if iscell(kLessDistant)
                        if iscellstr(kLessDistant)
                            %maxscore = mode(char(kLessDistant));
                            maxscore = obj.strVoting(kLessDistant);
                        else
                            maxscore = mode(cell2mat(kLessDistant));
                        end
                    else
                        maxscore = mode(kLessDistant);
                    end
                end
                
                timeElapsed = toc;
            end
        end
        
    end
    
