function [mu sigma2] = estimateGaussian(X)

% Useful variables
[m, n] = size(X);

mu = zeros(n, 1);
sigma2 = zeros(n, 1);

% sum of rows
mu = sum(X,1)' ./ m;

sigma2 = sum(((X - mu') .^ 2), 1)' ./ m;

% =============================================================


end
