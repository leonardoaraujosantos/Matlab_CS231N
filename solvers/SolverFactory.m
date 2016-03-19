classdef SolverFactory < handle
    % Class that will create a solver        
    methods(Static)
        % Static method used just to shuffle data
        function solver = get(metaDataSolver)
            switch metaDataSolver.type
                case SolverType.GradientDescent
                    solver = GradientDescent(metaDataSolver.learningRate, metaDataSolver.numEpochs);
                case SolverType.StochasticGradientDescent
                    solver = StochasticGradientDescent(metaDataSolver.learningRate, metaDataSolver.batchSize, metaDataSolver.numEpochs);
                case SolverType.AdaptiveGradient
                    solver = AdaptiveGradient(metaDataSolver.learningRate, metaDataSolver.batchSize, metaDataSolver.numEpochs);
                case SolverType.NesterovAcceleratedGradient
                    solver = NesterovAcceleratedGradient(metaDataSolver.learningRate, metaDataSolver.batchSize, metaDataSolver.numEpochs);
            end 
        end
    end    
end

