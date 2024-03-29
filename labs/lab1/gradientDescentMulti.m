function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

timeStart = cputime;
for iter = 1:num_iters
    h =  X * theta;    
    theta = theta - alpha * (1/m) * ((h - y)' * X)';
    J_history(iter) = computeCostMulti(X, y, theta);
end
timeEnd = cputime;
fprintf(['Processing time in seconds: %f \n'], timeEnd-timeStart);
end


