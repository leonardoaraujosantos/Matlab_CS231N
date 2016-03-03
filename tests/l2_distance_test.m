import matlab.unittest.constraints.IsEqualTo;

%% Test 1: Check numeric example (Normal array)
I1 = [56 32 10 18; 90 23 128 133; 24 26 178 200; 2 0 255 220];
I2 = [10 20 24 17; 8 10 89 100; 12 16 178 170; 4 32 233 112];

% Squeeze 2d vector to 1d
I1 = reshape(I1',[1,numel(I1)]);
I2 = reshape(I2',[1,numel(I2)]);

% Calculate L2 distance (Euclidian)
distance = l2_distance(I1,I2);

% Check output
error = abs(distance - norm(I1-I2));
assert (error < 0.0001); 

%% Test 2: Check numeric example (gpuarray array)
I1 = [56 32 10 18; 90 23 128 133; 24 26 178 200; 2 0 255 220];
I2 = [10 20 24 17; 8 10 89 100; 12 16 178 170; 4 32 233 112];

% Squeeze 2d vector to 1d
I1 = reshape(I1',[1,numel(I1)]);
I2 = reshape(I2',[1,numel(I2)]);

% Convert to gpuArray (send data to gpu)
I1 = gpuArray(I1);
I2 = gpuArray(I2);

% Calculate L2 distance (Euclidian)
distance = l2_distance(I1,I2);
% Get data from GPU
distance = gather(distance); %Should be 456

% Check output
error = abs(distance - gather(norm(I1-I2)));
assert (error < 0.0001); 

%% Test 3: Check big array on GPU
I1 = gpuArray(rand(1,20000000));
I2 = gpuArray(rand(1,20000000));
sizeArray = numel(I1);

% Calculate L2 distance (Euclidian)
tic;
distance = l2_distance(I1,I2);
% Get data from GPU
distance = gather(distance);
elapsed = toc;
fprintf('L2 dist) Took %d secs on %d elements GPU\n',elapsed, sizeArray);

% Check output
error = abs(distance - gather(norm(I1-I2)));
assert (error < 0.0001); 

%% Test 4: Check big array on CPU
I1 = (rand(1,20000000));
I2 = (rand(1,20000000));
sizeArray = numel(I1);

% Calculate L2 distance (Euclidian)
tic;
distance = l2_distance(I1,I2);
% Get data from GPU
distance = (distance);
elapsed = toc;
fprintf('L2 dist) Took %d secs on %d elements CPU\n',elapsed, sizeArray);

% Check output
error = abs(distance - (norm(I1-I2)));
assert (error < 0.0001); 