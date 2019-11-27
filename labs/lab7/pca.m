function [U, S] = pca(X)

% Useful values
[m, n] = size(X);

U = zeros(n);
S = zeros(n);

Sigma = 1/ m * X' * X;

[U, S, V] = svd(Sigma);

% =========================

end
