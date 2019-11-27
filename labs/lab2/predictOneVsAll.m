function p = predictOneVsAll(all_theta, X)

m = size(X, 1);
num_labels = size(all_theta, 1);

p = zeros(size(X, 1), 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];  


sigmoid_data = sigmoid(X * all_theta');
[max_elem, p(:,1)] = max(sigmoid_data,[],2);

end