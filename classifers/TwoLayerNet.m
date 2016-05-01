classdef TwoLayerNet < handle
    %TWOLAYERNET Simple toy network with 1
    %   Detailed explanation goes here
    
    properties
        params
        regularization
        lossFunc
        
        fcLayer
        reluLayer
        fcLayer_out
    end
    
    methods
        function obj = TwoLayerNet(input_dim, hidden_dim, numClasses)
            obj.regularization = 0;
            
            % Matlab does not have a dictionary the most similar concept is
            % a map
            % http://uk.mathworks.com/help/matlab/ref/containers.map-class.html
            W1 = rand(input_dim,hidden_dim);
            B1 = zeros(1,hidden_dim);
            W2 = rand(hidden_dim,numClasses);
            B2 = zeros(1,numClasses);
            
            obj.params = {W1,W2,B1,B2};
            
            obj.lossFunc = SoftMaxLoss();
            
            obj.fcLayer = InnerProductLayer();
            obj.reluLayer = ReluLayer();
            obj.fcLayer_out = InnerProductLayer();
        end
        
        function numParams = getParamCount(obj)
            numParams = size(obj.params,2);
        end
        
        function [loss, grads, scores] = loss(obj, X_vec, varargin)
            % Forward Pass
            z1 = obj.fcLayer.feedForward(X_vec,obj.params{1},obj.params{3});
            a2 = obj.reluLayer.feedForward(z1);
            z2 = obj.fcLayer_out.feedForward(a2,obj.params{2},obj.params{4});
            scores = z2;
            
            if (nargin < 3)            
                loss = 0;                
                grads = [];
                return;
            end
            Y_vec = varargin{1};
            
            % Back propagation
            [loss, dout] = obj.lossFunc.getLoss(z2,Y_vec);
            [delta2,dW2,dB2] = obj.fcLayer_out.backPropagate(dout);
            reluDelta2 = obj.reluLayer.backPropagate(delta2);
            [delta1,dW1,dB1] = obj.fcLayer.backPropagate(reluDelta2);
            
            W1 = obj.params{1};
            W2 = obj.params{2};
            
            % Add the regularization effect
            dW1 = dW1 + (W1 * obj.regularization);
            dW2 = dW2 + (W2 * obj.regularization);
            
            % Return the partial derivatives
            grads = {dW1,dW2,dB1,dB2};
            
            % Add the regulizarization effect to the loss
            sumW1Sqrd = sum(W1(:).*W1(:));
            sumW2Sqrd = sum(W2(:).*W2(:));
            loss = loss + (0.5 * obj.regularization * (sumW1Sqrd + sumW2Sqrd));
        end
    end
end

