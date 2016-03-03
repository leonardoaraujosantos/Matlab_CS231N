%% Test 1: Initialization
sysExp = SystemCapabilitiesExplorer();

%% Test 2: Check Accept Double method
sysExp = SystemCapabilitiesExplorer();
result = sysExp.acceptDoubleGPU();
fprintf('Accept double on GPU %d\n', result);

%% Test 3: Check Computing capabilities
sysExp = SystemCapabilitiesExplorer();
result = sysExp.computeCapabilityGPU();
fprintf('GPU Compute capability %d\n', result);

%% Test 4: Check Cluster available
sysExp = SystemCapabilitiesExplorer();
result = sysExp.getClustersAvailable();
disp(result);

%% Test 5: Get number of local workers
sysExp = SystemCapabilitiesExplorer();
result = sysExp.getNumLocalWorkers();
fprintf('Number of local workers %d\n', result);

%% Test 6: Check OS
sysExp = SystemCapabilitiesExplorer();
result = sysExp.getOsInfo();
fprintf('Number of local workers %s\n', result);

%% Test 7: Check isWhat
sysExp = SystemCapabilitiesExplorer();
result = sysExp.isWhat();
fprintf('Number of local workers %s\n', result);
