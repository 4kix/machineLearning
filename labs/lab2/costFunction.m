function [J, grad] = costFunction(theta, X, y)

m = length(y); % number of training examples

J = 0;
grad = zeros(size(theta));
h = sigmoid(X*theta);

% J(theta)
J = 1/m * (-y' * log(h) - (1 - y)' * log(1-h)); 

%  gradient descent
grad = 1/m * X' * (h - y)

end

