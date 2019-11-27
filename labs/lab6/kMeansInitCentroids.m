function centroids = kMeansInitCentroids(X, K)

% You should return this values correctly
centroids = zeros(K, size(X, 2));

% Random reorder the indeces of examples
randidx = randperm(size(X,1));
% Take the first K examples as centroids
centroids = X(randidx(1:K), :);

% =============================================================

end

