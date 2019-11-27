function centroids = computeCentroids(X, idx, K)

% Useful variables
[m n] = size(X);

centroids = zeros(K, n);

for i = 1:K
  pos = find(idx == i);
  
  % sum on columns
  centroids(i,:) = sum(X(pos,:),1) ./ size(pos,1);

endfor


% =================================================


end

