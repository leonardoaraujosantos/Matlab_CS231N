function [ f ] = forwardCircuit( x,y,z )
    q = forwardAddGate(x,y);
    f = forwardMultiplyGate(q,z);
end

function [q] = forwardAddGate(a,b)
    q = a + b;
end

function [z] = forwardMultiplyGate(a,b)
    z = a*b;
end

