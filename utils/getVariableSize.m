function [ outSize ] = getVariableSize( variable, varargin )
%GETVARIABLESIZE Get the size of a variable in bytes,megaBytes,gigabytes...

% InputName(1) gets the variable name of the first parameter
%variableStringName = inputname(1);
% Trick: On the function stack variable is 'variable' so no need to get the
% input argument variable name
sizeBytes = whos('variable');

numArguments = nargin;
if numArguments > 1
    parMetric = varargin{1};
else
    parMetric = 0;
end

% 0(Bytes), 1(kBytes), 2(MegaBytes), 3(GigaBytes)
outSize = (sizeBytes.bytes) / (1024^parMetric);

end

