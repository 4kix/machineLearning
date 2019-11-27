function [J, grad] = linearRegCostFunction(X, y, theta, lambda)

m = length(y); % number of training examples

J = 0;
grad = zeros(size(theta));

% J

coef = 1 / (2 * m);
reg_coef = lambda / (2 * m);
square_sum = sum((X * theta - y) .^ 2);
theta_part = sum(theta(2:end) .^ 2);
J =  coef * (square_sum) + reg_coef * theta_part;

% gradient descent

grad_coef = 1 /  m;
grad_reg_coef = lambda / m;
use_coef_reg = ones(size(theta),1);
use_coef_reg(1) = 0;
grad = grad_coef * (sum(( X * theta - y) .* X))' .+ ...
                        ( grad_reg_coef .* theta .* use_coef_reg );


grad = grad(:);

end

