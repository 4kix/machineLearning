function idx = findClosestCentroids(X, centroids)

% Set K
K = size(centroids, 1);

idx = zeros(size(X,1), 1);

for i = 1:size(idx,1)
  curr_elem = X(i,:);
  distance = zeros(K,1);
  
  
  for j = 1:K
    distance(j) = sum((centroids(j,:) .- curr_elem) .^ 2);
  endfor
  
  [temp, idx(i)] = min(distance);
  
endfor

% =============================================================

end

