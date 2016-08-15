function [ reward, state ] = simple_RL_enviroment( action, restart )
% Simple enviroment of reinforcement learning example
%   http://mnemstudio.org/path-finding-q-learning-tutorial.htm
persistent current_state
if isempty(current_state)
    % Initial random state (excluding goal state)
    current_state = randi([1,5]);     
end

% The rows of R encode the states, while the columns encode the action
R = [ -1 -1 -1 -1  0  -1; ...
    -1 -1 -1  0 -1 100; ...
    -1 -1 -1  0 -1  -1; ...
    -1  0  0 -1  0  -1; ...
    0 -1 -1  0 -1 100; ...
    -1  0 -1 -1  0 100 ];

% Sample our R matrix (model)
reward = R(current_state,action);

% Good action taken
if reward ~=-1
    % Returns next state (st+1)
    current_state = action;        
end

% Game should be reseted
if restart == true
    % Choose another initial state
    current_state = randi([1,5]); 
    reward = -1;
    % We decrement 1 because matlab start the arrays at 1, so just to have
    % the messages with the same value as the figures on the tutorial we
    % take 1....
    fprintf('Enviroment initial state is %d\n',current_state-1);
end

state = current_state;

end

