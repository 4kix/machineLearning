function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

timeStart = cputime;

for iter = 1:num_iters

    coef = alpha / m;

    theta_res = [];
    theta_size = length(theta);
    for iter_theta = 1:theta_size

        x_clmn = X(:,iter_theta);
        theta_clmn = theta(iter_theta,:);
        temp_sum = sum((X * theta - y) .* x_clmn);
        theta_clmn = theta_clmn - (coef * temp_sum);
        theta_res = [theta_res; theta_clmn];

    end

    theta = theta_res;

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

timeEnd = cputime;
fprintf(['Processing time in seconds: %f \n'], timeEnd-timeStart);

end
