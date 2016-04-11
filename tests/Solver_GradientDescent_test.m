%% Gradient descent
% Iterative algorithm that minimize a function (Find it's global mimima),
% it's a greedy algorithm, which means that it will stuck on the local
% minima on the case of a non-convex function.
% It's used on deep neural networks though because when you have a lot of
% layers the local minima will have similar training error as the global
% minima.
%
% http://en.wikipedia.org/wiki/Gradient_descent
%
% Code:
%
% <include>GradientDescent.m</include>
%

%% Test 1: Simple function minimization 
% $f(x) = x^4-3x^3+2$
%
% ${\frac{df(x)}{dx}\ = 4x^3-9x^2}$
%
%% Use matlab symbolic engine to get the derivatives
% http://www.mathworks.com/help/symbolic/find-asymptotes-critical-and-inflection-points.html
% Define symbol x
syms x;
% Define function f(x)
f_x = x^4-3*x^3+2;
pretty(f_x);

% Calculate derivative d(f(x))/dx
deriv_f_x = diff(f,x);
pretty(deriv_f_x);

% Look for critical points (Solve derivative equal to zero)
crit_pts = solve(deriv_f_x);
pretty(crit_pts);

% Plot function
fplot(matlabFunction(f_x),[-2 3.5]);
title('Plot function and critical points (minimas,maximas) if available');
hold on
plot(double(crit_pts), double(subs(f_x,crit_pts)),'ro');
text(0,5,'Local minima');
text(2.3,-4,'Global minima');
axisPlot = gca; % current axes
%% Generate anonymous matlab function from symbolic expression
% Doing by hand: f_derivative = @(x) 4*x.^3 - 9*x.^2;
%
% http://www.mathworks.com/help/matlab/matlab_prog/anonymous-functions.html?refresh=true
%
% http://www.mathworks.com/help/symbolic/generate-matlab-functions.html
%
f_derivative = matlabFunction(deriv_f_x);
f_x_numeric = matlabFunction(f_x);

%% Using gradient descent solver
% Starting at x=1 (Don't start with zero)
x_new = 0.5; % The algorithm starts at x=1
% If we start from x=-1 we will stuck on a local minima (=~0)

% Create solver of type Gradient descent
solver = SolverFactory.get(struct('type',SolverType.GradientDescent,'learningRate',0.01, 'numEpochs', 500));

% Optimize
x_new_vec = zeros(1,solver.epochs);
y_new_vec = zeros(1,solver.epochs);
for idxEpochs = 1:solver.epochs    
    delta = f_derivative(x_new);
    % Observe that the optimization just need the current parameters and
    % the derivative of the function evaluated on the current parameter
    x_new = solver.optimize(x_new,delta);
    %x_new = x_old - gamma * f_derivative(x_old);
    
    % Used to debug the gradient descent
    x_new_vec(idxEpochs) = x_new;
    y_new_vec(idxEpochs) = f_x_numeric(x_new);
end
correct_result = 9/4;
error = abs(x_new - correct_result);
fprintf('Local minimum found at %d and should be %d error=%d\n',...
    x_new,correct_result,error);
comet(axisPlot,x_new_vec,y_new_vec);
hold off

%% Test 2: Get stuck on local minima
% $f(x) = x^4-3x^3+2$
%
% ${\frac{df(x)}{dx}\ = 4x^3-9x^2}$
%
%% Use matlab symbolic engine to get the derivatives
% http://www.mathworks.com/help/symbolic/find-asymptotes-critical-and-inflection-points.html
% Define symbol x
syms x;
% Define function f(x)
f_x = x^4-3*x^3+2;
pretty(f_x);

% Calculate derivative d(f(x))/dx
deriv_f_x = diff(f,x);
pretty(deriv_f_x);

% Look for critical points (Solve derivative equal to zero)
crit_pts = solve(deriv_f_x);
pretty(crit_pts);

% Plot function
fplot(matlabFunction(f_x),[-2 3.5]);
title('Plot function and critical points (minimas,maximas) if available');
hold on
plot(double(crit_pts), double(subs(f_x,crit_pts)),'ro');
text(0,5,'Local minima');
text(2.3,-4,'Global minima');
axisPlot = gca; % current axes

%% Generate anonymous matlab function from symbolic expression
% Doing by hand: f_derivative = @(x) 4*x.^3 - 9*x.^2;
%
% http://www.mathworks.com/help/matlab/matlab_prog/anonymous-functions.html?refresh=true
%
% http://www.mathworks.com/help/symbolic/generate-matlab-functions.html
%
f_derivative = matlabFunction(deriv_f_x);

%% Using gradient descent solver
% Starting at x=-2 Don't start with zero, on this case we can stuck on a
% local minima at x=0
x_new = -2;

% Create solver of type Gradient descent
solver = SolverFactory.get(struct('type',SolverType.GradientDescent,'learningRate',0.01, 'numEpochs', 500));

% Optimize
x_new_vec = zeros(1,solver.epochs);
y_new_vec = zeros(1,solver.epochs);
for idxEpochs = 1:solver.epochs    
    delta = f_derivative(x_new);
    % Observe that the optimization just need the current parameters and
    % the derivative of the function evaluated on the current parameter
    x_new = solver.optimize(x_new,delta);
    %x_new = x_old - gamma * f_derivative(x_old);
    
    % Used to debug the gradient descent
    x_new_vec(idxEpochs) = x_new;
    y_new_vec(idxEpochs) = f_x_numeric(x_new);
end
comet(axisPlot,x_new_vec,y_new_vec);
hold off

correct_result = 9/4;
error = abs(x_new - correct_result);
fprintf('Local minimum found at %d and should be %d error=%d\n',...
    (x_new),correct_result,error);