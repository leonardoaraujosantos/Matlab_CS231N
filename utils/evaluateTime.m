function [ timeSpent ] = evaluateTime( inMatrixSize, isDistributed )
%EVALUATETIME Just to a heavy operation and return the execution time
% evaluateTime(4096*2*2*2,false)
if isDistributed
    A = rand(inMatrixSize,inMatrixSize,'distributed');
    B = rand(inMatrixSize,inMatrixSize,'distributed');
else
    A = rand(inMatrixSize,inMatrixSize);
    B = rand(inMatrixSize,inMatrixSize);
end

tic; 
C = (A\B)*A*B; 
timeSpent = toc;

end

