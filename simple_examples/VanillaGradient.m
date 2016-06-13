% Define function symbolically
syms w;
L_w = w^4-3*w^3+2;

% Get numeric version of loss function
L_w_mat = matlabFunction(L_w);

% Calculate derivative of loss related to w
deriv_f_w = diff(L_w,w);

% Get numeric version of derivative of loss function
f_derivative = matlabFunction(deriv_f_w);

% Symbolic find the critical points (Solve derivative equal to zero)
crit_pts = solve(deriv_f_w);
pretty(round(crit_pts));

% Initial weight value
weight = -2;

% Step size
step = 0.01;

% Plot function
fplot(matlabFunction(L_w),[-2 3.5]);
title('Plot function and critical points (minimas,maximas) if available');
hold on
plot(double(crit_pts), double(subs(L_w,crit_pts)),'ro');
text(0,5,'Local minima');
text(2.3,-4,'Global minima');
axisPlot = gca; % current axes

% Add dot to the initial position
redDot = plot(weight,L_w_mat(weight),'o','MarkerFaceColor','red');
hold off
axis manual

% Gradient descent
for iters=1:500
    % Get gradient by evaluating derivative of Loss related to W
    weight_grad = f_derivative(weight);
    weight = weight - (step*weight_grad);
        
    % Animate dot
    redDot.XData = weight;
    redDot.YData = L_w_mat(weight);
    drawnow limitrate
    %pause(0.3)
end

hold off;