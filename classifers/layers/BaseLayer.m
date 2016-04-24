classdef (Abstract) BaseLayer < handle
    %BASELAYER Abstract class for Layer
    % Check format on reference project:
    % http://cs.stanford.edu/people/karpathy/convnetjs/
    % https://github.com/karpathy/convnetjs 
    % https://databoys.github.io/Feedforward/
    % http://scs.ryerson.ca/~aharley/neural-networks/
    
    properties (Abstract)
        typeLayer
        weights
        activations
    end
    
    methods(Abstract, Access = public)
        [result] = feedForward(obj, inputs);
        
        % For the output layer
        % 1. Calculates the difference between output value and target value
        % 2. Get the derivative (slope) of the activation function in 
        %    order to determine how much the weights need to change
        % 3. update the weights for every node based on the learning rate 
        %    and activation function derivative
        
        %  For the hidden layer
        % 1. Calculate the sum of the strength of each output link 
        %    multiplied by how much the target node has to change
        % 2. Get derivative to determine how much weights need to change
        % 3. Change the weights based on learning rate and derivative
        [gradient] = backPropagate(obj);
        
        % Return any other data (ie: Activations) different from the normal
        % feedForward result
        [result] = getActivations(obj);
        
        % Return the layer type
        [type] = getType(obj); 
        
        % Get text description
        [descText] = getDescription(obj);
        
        % Get number of neurons
        [numNeurons] = getNumNeurons(obj);
        
        % Get number of parameters
        [numParameters] = getNumParameters(obj);
    end
    
end

