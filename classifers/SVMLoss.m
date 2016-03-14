classdef SVMLoss < BaseLossFunction
    % SVM Loss implementation class
    
    properties (Access = 'private')
        delta;
    end
    
    methods
        function obj = SVMLoss(parDelta)
            % Initialize SVM loss hyper-parameter (Bad one more thing to
            % think...)
            obj.delta = parDelta;
        end
        
        % Here the smalles value will be zero (Perfect, no loss) and the
        % biggest value is unbounded
        function [lossResult] = getLoss(obj, scores, idxCorrect)
                        
            % Get the correct score
            correct_score = scores(idxCorrect);
            
            % Get all scores except the wrong one
            % http://stackoverflow.com/questions/19596268/select-all-elements-except-one-in-a-vector-matlab
            incorrect_scores = scores(1:end ~= idxCorrect);
            lossResult = sum(max(0,(incorrect_scores - correct_score) + obj.delta));
                                    
        end
    end
    
end

