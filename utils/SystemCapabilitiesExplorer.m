classdef SystemCapabilitiesExplorer < handle
    %SYSTEMGPUEXPLORER Summary of this class goes here
    %   Detailed explanation goes here
    
    properties ( Access = 'private' )
        numOfGpus;
        gpuVec;
        numWorkers;
        localCluster;
        clustersAvailable;
        operationSystemDesc;
    end
    
    methods ( Access = 'public' )
        function obj = SystemCapabilitiesExplorer()
            % Get number of gpus
            obj.numOfGpus = gpuDeviceCount();
            
            % Initialize gpuVector
            obj.gpuVec = parallel.gpu.CUDADevice();
            for idxGpu=1:obj.numOfGpus
                obj.gpuVec(idxGpu) = gpuDevice(idxGpu);
            end
            
            obj.localCluster = parcluster('local');
            obj.clustersAvailable = parallel.clusterProfiles;
            
            % http://undocumentedmatlab.com/blog/undocumented-feature-function
            obj.operationSystemDesc = feature('GetOS');
        end
        
        function fAccept = acceptDoubleGPU(obj,varargin)
            if nargin == 2
                % Passing one parameter (index)
                fAccept = obj.gpuVec(varargin{1}).SupportsDouble;
            else
                % Passing no parameter
                fAccept = obj.gpuVec(1).SupportsDouble;
            end
        end
        
        function compCap = computeCapabilityGPU(obj,varargin)
            if nargin == 2
                % Passing one parameter (index)
                compCap = str2num(obj.gpuVec(varargin{1}).ComputeCapability);
            else
                % Passing no parameter
                compCap = str2num(obj.gpuVec(1).ComputeCapability);
            end
        end
        
        function clusterDesc = getClustersAvailable(obj)
            clusterDesc = obj.clustersAvailable;
        end
        
        function numWorkers = getNumLocalWorkers(obj)
            numWorkers = obj.localCluster.NumWorkers;
        end
        
        function osDescription = getOsInfo(obj)
            osDescription = obj.operationSystemDesc;
        end
        
        function sys_info = isWhat(vargin)
            if ismac
                % Code to run on Mac plaform
                sys_info = 'is_mac';
            elseif isunix
                % Code to run on Linux plaform
                sys_info = 'is_linux';
            elseif ispc
                % Code to run on Windows platform
                sys_info = 'is_windows';
            else
                sys_info = 'is_unknown';
            end
        end
    end
    
end

