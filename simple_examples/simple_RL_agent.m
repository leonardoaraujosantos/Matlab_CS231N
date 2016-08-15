function [ Q ] = simple_RL_agent( )
% Simple agent of reinforcement learning example
%   http://mnemstudio.org/path-finding-q-learning-tutorial.htm
% Train, then normalize Q (divide Q by it's biggest value)
Q = train(); Q = Q / max(Q(:));
% Get the best actions for each possible initial state (1,2,3,4,5)
test(Q);
end

function Q = train()
% Initial training parameters
gamma = 0.8;
goalState=6;
numTrainingEpisodes = 20;
% Set Q initial value
Q = zeros(6,6);

% Learn from enviroment iteraction
for idxEpisode=1:numTrainingEpisodes
    validActionOnState = -1;
    % Reset environment
    [~,currentState] = simple_RL_enviroment(1, true);
    
    % Episode (initial state to goal state)
    % Break only when we reach the goal state
    while true
        % Choose a random action possible for the current state
        while validActionOnState == -1
            % Select a random possible action
            possibleAction = randi([1,6]);
            
            % Interact with enviroment and get the immediate reward
            [ reward, ~ ] = simple_RL_enviroment(possibleAction, false);
            validActionOnState = reward;
        end
        validActionOnState = -1;
                
        % Update Q
        % Get the biggest value from each row of Q, this will create the
        % qMax for each state
        next_state = possibleAction;
        qMax = max(Q,[],2);
        Q(currentState,possibleAction) = reward + ...
            (gamma*(qMax(next_state)));
                
        if currentState == goalState
            break;
        end
        
        % Non this simple example the next state will be the action
        currentState = possibleAction;
    end
    fprintf('Finished episode %d restart enviroment\n',idxEpisode);
end
end

function test(Q)
    % Possible permuted initial states, observe that you don't include the
    % goal state 6 (room5)
    possible_initial_states = randperm(5);
    goalState=6;
    
    % Get the biggest action for every state
    [~, action_max] = max(Q,[],2);
        
    for idxStates=1:numel(possible_initial_states)
        curr_state = possible_initial_states(idxStates);
        fprintf('initial state room_%d actions=[ ', curr_state-1);
        % Follow optimal policy from intial state to goal state
        while true
            next_state = action_max(curr_state);
            fprintf(' %d,', next_state-1);
            curr_state = next_state;
            if curr_state == goalState
                fprintf(']');
               break 
            end
        end
        fprintf('\n');
    end
end

