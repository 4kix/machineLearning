function J = computeCostMulti(X, y, theta)
  
m = length(y); % number of training examples

J = 0;

predictions = X * theta;
squareErr = (predictions - y).^2;
J = 1 / (2 * m) * sum(squareErr);


end
