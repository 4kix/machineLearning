function [J, grad] = costFunctionReg(theta, X, y, lambda)

m = length(y); % number of training examples

J = 0;
grad = zeros(size(theta));

coef = 1 / m;
coef_reg = lambda / (2 * m);
coef_reg_grad = lambda / m;
use_coef_reg = ones(size(theta),1);
use_coef_reg(1) = 0;

% J(theta)
temp_vect = 2:1:size(theta);

temp_sum = sum(-y .* log(sigmoid(X * theta)) .- ...
              ((1 - y) .* log(1 - sigmoid(X * theta))));
J =  coef * (temp_sum) + coef_reg * sum(theta(temp_vect, :) .^ 2);

% gradient descent
grad = coef * (sum(( sigmoid(X * theta) .- y) .* X))' .+ ...
        ( coef_reg_grad .* theta .* use_coef_reg );

end
