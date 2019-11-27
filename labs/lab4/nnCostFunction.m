function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)

                                   
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% Prepare Y data (map value)

y_new = zeros(size(y), size(Theta2,1));
for i = 1:m
  y_new(i, y(i)) = 1;
end;

%	Feedforward

a1 = X;
z2 = [ones(m, 1) X] * Theta1';
a2 = sigmoid(z2);
z3 = [ones(m, 1) a2] * Theta2';
a3 = sigmoid(z3);

%	Calculation J

for i = 1:m
  y_temp = y_new(i,:);
  a3_temp = a3(i,:);
  temp_sum = sum(-y_temp .* log(a3_temp) .- ((1 - y_temp) .* log(1 - a3_temp)));
  J = J + temp_sum;
end;

coef = 1 / m;
J = coef * J;


% Calculation J & Regularization

reg_coef = lambda / (2 * m);
J = J + reg_coef *( sum(sum(Theta1(:, 2:end) .^ 2)) + sum(sum(Theta2(:, 2:end) .^ 2)) );

% Calculation Theta1_grad & Theta2_grad


delta2 = zeros(m, size(Theta1, 1)); % 5000 x 25
delta3 = zeros(m, size(Theta2, 1)); % 5000 x 10

for i = 1:m
	predict_value = a3(i,:);
	y_value = y_new(i,:);
	
  delta3_temp = predict_value .- y_value;
  delta3(i,:) = delta3_temp';

end

delta2 = delta3 * Theta2(:, 2:end) .* (sigmoidGradient(z2));

DELTA_1 = delta2' * [ones(m, 1) a1]; % 25 x 401
DELTA_2 = delta3' * [ones(m, 1) a2]; % 10 x 26

Theta1_grad = coef .* DELTA_1;
Theta2_grad = coef .* DELTA_2;

% Add regularization

Theta1_grad(:, 2:end) = Theta1_grad(:, 2:end) .+ (lambda / m) .* (Theta1(:, 2:end));
Theta2_grad(:, 2:end) = Theta2_grad(:, 2:end) .+ (lambda / m) .* (Theta2(:, 2:end));

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
