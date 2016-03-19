%% Activation Functions
% In computational networks, the activation function of a node defines the 
% output of that node given an input or set of inputs.
%
% First the neuron will do a dot product between the inputs and it's
% weights, then this result is feed to a activation function
% 
% <</home/leo/work/Matlab_CS231N/docs/imgs/ArtificialNeuronZoom.jpg>>
%
% 
% Complete table:
% 
% <</home/leo/work/Matlab_CS231N/docs/imgs/TableCompleteActivationFunctions.png>>
%
% More info at:
% https://en.wikipedia.org/wiki/Activation_function
% http://ufldl.stanford.edu/wiki/index.php/Neural_Networks

%% Test 1: Test Sigmod
% $f(x) = \frac{1}{(1+e^{x})}$
% 
% <include>SigmoidActivation.m</include>
%
X = -10:0.1:10;
outFunc = SigmoidActivation.forward_prop(X);
hFig = figure(1);
set(hFig, 'Position', [0 0 800 500])
subplot(1,2,1);
plot(X,outFunc);
title('Sigmoid');
xlabel('X');
ylim([0 1]);
subplot(1,2,2);
d_outFunc = SigmoidActivation.back_prop(X);
plot(X,d_outFunc);
title('Backprop Sigmoid');

%% Test 2: Test Tanh
% $f(x) = \frac{2}{(1+e^{-2x})}-1$
% 
% <include>TanhActivation.m</include>
%
X = -10:0.1:10;
outFunc = TanhActivation.forward_prop(X);
hFig = figure(1);
set(hFig, 'Position', [0 0 800 500])
subplot(1,2,1);
plot(X,outFunc);
title('Tanh');
xlabel('X');
ylim([-1 1]);
subplot(1,2,2);
d_outFunc = TanhActivation.back_prop(X);
plot(X,d_outFunc);
title('Backprop Tanh');

%% Test 3: Test Relu
% $f(x) = max(0,x)$
% 
% <include>ReluActivation.m</include>
%
X = -10:0.1:10;
outFunc = ReluActivation.forward_prop(X);
hFig = figure(1);
set(hFig, 'Position', [0 0 800 500])
subplot(1,2,1);
plot(X,outFunc);
title('Relu');
xlabel('X');
xlim([-10 10]);
ylim([-0.1 10]);
subplot(1,2,2);
d_outFunc = ReluActivation.back_prop(X);
plot(X,d_outFunc);
title('Backprop Relu');