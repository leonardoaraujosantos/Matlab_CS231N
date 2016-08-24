function [ output_args ] = spmdExample( numIterations, poolWokers )
%SPMDEXAMPLE Simple example of spmd usage implementing SEDA architecture
%   Detailed explanation goes here

output_args = 0;

% Tags for identify the data
tagData = 10;
tagProcData = 20;
tagFinish = 666;

% Workers indexes
numWorkers = poolWokers-2;
workerIndexStart = 2;

% Variable that holds which worker completely finish it's job
workerActivity = zeros(numWorkers,1);

% Parallel mode here....
spmd(poolWokers)
    if labindex == 1
        fprintf('Num workers available:%d\n',numWorkers);
        % Distribute "jobs" to workers
        for idxJob=1:numIterations
            % Round robin
            for idx=1:numWorkers
                % Get data from somewhere far far away....
                someData = rand() * 10;
                % Calculate the labIndex
                slaveCode = (workerIndexStart+idx)-1;
                fprintf('Send data <%d> to worker: %d\n',someData,slaveCode);
                labSend(someData,slaveCode,tagData);
            end
        end
        fprintf('Job dispatched to all workers, now wait....\n');
        for idxLabs=2:poolWokers-1
            labSend(0,idxLabs,tagFinish);
        end
        
        % Bad way to ask the aggregator process to stop because there may have
        % some items on the workers to finish.
        %pause(10);
        %labSend(0,poolWokers,tagFinish);
    else
        % Workers
        if (labindex >= 2) && (labindex <=numWorkers+1)
            while true
                % Wait for data available from lab1
                if labProbe(1)
                    [data, ~, tag] = labReceive(1);
                    if tag == tagFinish
                        fprintf('Ask to stop\n');
                        break;
                    end
                    fprintf('Received data %f\n',data);
                    % Very long process that could take random time to finish
                    dataProc = labindex;
                    pause(rand()*2);
                    
                    % Send calculation to aggregation process
                    labSend(dataProc,poolWokers,tagProcData);
                end
            end
            % Signal aggregation that this worker has finished
            labSend(0,poolWokers,tagFinish);
        else
            if labindex == poolWokers
                while true
                    % Wait for data available
                    if labProbe()
                        [data, sourceLab, tag] = labReceive;
                        if tag == tagFinish
                            fprintf('AGGR: Ask to stop from lab %d\n',sourceLab);
                            workerActivity(sourceLab-1) = 1;
                            % Stop when all workers finished
                            if all(workerActivity)
                                fprintf('AGGR: Stop now\n');
                                break;
                            end
                        end
                        fprintf('AGGR: Received data %f\n',data);
                    end
                end
            end
        end
    end
end

end

