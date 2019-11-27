function [error_train, error_val] = ...
    learningCurve(X, y, Xval, yval, lambda)

m = size(X, 1);

error_train = zeros(m, 1);
error_val   = zeros(m, 1);

for i = 1:m
	X_temp = X(1:i, :);
	y_temp = y(1:i);

	train_theta = trainLinearReg(X_temp, y_temp, lambda);
	error_train(i) = (1 / (2 * i) ) * sum((X_temp * train_theta .- y_temp) .^ 2);
	error_val(i) = (1 / (2 * size(Xval, 1))) * sum((Xval * train_theta .- yval) .^ 2);
end

end

