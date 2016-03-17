classdef SolverType < uint32
    % Define supported solvers available on this system
    enumeration
      GradientDescent (1)
      StochasticGradientDescent  (2)
      AdaptiveGradient  (3)
      NesterovAcceleratedGradient (4)      
    end    
end
