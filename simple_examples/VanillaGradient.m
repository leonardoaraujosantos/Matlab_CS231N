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
weight = -1.5;

% Step size
step = 0.0001;

% Plot function
filename = 'gradDescentAnim.gif';
figure(1);
fplot(matlabFunction(L_w),[-2 3.5]);
title('L(w)=w^4-3w^3+2');
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
for iters=1:20
    %% Get gradient by evaluating derivative of Loss related to W    
    weight_grad = f_derivative(weight);
    weight = weight - (step*weight_grad);
    
    % Animate dot
    redDot.XData = weight;
    redDot.YData = L_w_mat(weight);
    drawnow limitrate
        
    % Create animated gif
    frame = getframe(1);
    im = frame2im(frame);
    [imind,cm] = rgb2ind(im,256);
    if iters == 1;
        imwrite(imind,cm,filename,'gif', 'Loopcount',inf);
    else
        imwrite(imind,cm,filename,'gif','WriteMode','append');
    end
    
    % Just to make the animation more interesting
    step = step * 1.5;
end

hold off;