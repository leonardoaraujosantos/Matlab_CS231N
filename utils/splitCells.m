function [ outCell ] = splitCells( input, parts )
% Split cell in x parts
nElementsIn = numel(input);
numParts = floor(nElementsIn/parts);
remainder = mod(nElementsIn, parts);

idxInput = 1;
for idx=1:parts
    outCell{idx} = input(idxInput:(idxInput + numParts - 1));    
    
    % Add the remainder on the last part
    if (idx == parts)
        if (remainder ~= 0)
            outCell{idx} = input(idxInput:end); 
        end
    end
    
    idxInput = idxInput + numParts;
end

end

