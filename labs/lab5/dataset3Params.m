function [C, sigma] = dataset3Params(X, y, Xval, yval)

C = 1;
sigma = 0.3;

posible_value_vec = [0.01 0.03 0.1 0.3 1 3 10 30]

C_index = 1;
sigma_index = 1;
min_mean = 1;

for i = 1:length(posible_value_vec)
	for j = 1:length(posible_value_vec)
		temp_C = posible_value_vec(i);
		temp_sigma = posible_value_vec(j);

		model= svmTrain(X, y, temp_C, @(x1, x2) gaussianKernel(x1, x2, temp_sigma));
		predictions = svmPredict(model, Xval);
		curr_mean = mean(double(predictions ~= yval));

		if curr_mean < min_mean
			min_mean = curr_mean;
			C_index = i;
			sigma_index = j;
		end
	end
end

C = posible_value_vec(C_index);
sigma = posible_value_vec(sigma_index);

% =========================================================================

end
