function f = fibonacci_slow(n)

% A simple example for generating the N'th Fibonacci number 
% Uses the Matrix Exponential method

F = [1 1;1 0]^n;
f = F(1,2);

end