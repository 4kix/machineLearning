%% Initialization
clear ; close all; clc

%% ======================= Ex 1: Load =======================

fprintf('Loading data Data ...\n')
data = load('ex1data1.txt');

X = data(:, 1); y = data(:, 2);
m = length(y); % number of training examples

%% ======================= Ex 2: Plot =======================

plotData(X, y);
pause;

%% ======================= Ex 3,4:Cost Function and Gradient Descent ==========

X = [ones(m, 1), data(:,1)]; % Add a column of ones to x
theta = zeros(2, 1); % initialize fitting parameters

% Some gradient descent settings
iterations = 1500;
alpha = 0.01;

fprintf('\nRunning Gradient Descent ...\n')
% run gradient descent
theta = gradientDescent(X, y, theta, alpha, iterations);

% Plot the linear fit
hold on; % keep previous plot visible
plot(X(:,2), X*theta, '-')
legend('Training data', 'Linear regression')
hold off % don't overlay any more plots on this figure

pause;

%% ======================= Ex 5: Visualizing J(theta_0, theta_1) =============

fprintf('Visualizing J(theta_0, theta_1) ...\n')

% Grid over which we will calculate J
theta0_vals = linspace(-10, 10, 100);
theta1_vals = linspace(-1, 4, 100);

% initialize J_vals to a matrix of 0's
J_vals = zeros(length(theta0_vals), length(theta1_vals));

% Fill out J_vals
for i = 1:length(theta0_vals)
    for j = 1:length(theta1_vals)
	    t = [theta0_vals(i); theta1_vals(j)];
	    J_vals(i,j) = computeCost(X, y, t);
    end
end

% Because of the way meshgrids work in the surf command, we need to
% transpose J_vals before calling surf, or else the axes will be flipped
J_vals = J_vals';
% Surface plot
figure;
surf(theta0_vals, theta1_vals, J_vals)
xlabel('\theta_0'); ylabel('\theta_1');

% Contour plot
figure;
% Plot J_vals as 15 contours spaced logarithmically between 0.01 and 100
contour(theta0_vals, theta1_vals, J_vals, logspace(-2, 3, 20))
xlabel('\theta_0'); ylabel('\theta_1');
hold on;
plot(theta(1), theta(2), 'rx', 'MarkerSize', 10, 'LineWidth', 2);

pause;
%% ======================= Ex 6: Load ex1data2 =============

%% Close Figures
clear ;close all; 

fprintf('Loading data ...\n');

%% Load Data
data = load('ex1data2.txt');
X = data(:, 1:2);
y = data(:, 3);
m = length(y);

pause;

%% ======================= Ex 7: Feature Normalization =============


% Scale features and set them to zero mean
fprintf('Normalizing Features ...\n');

[X mu sigma] = featureNormalize(X);

% Add intercept term to X
X = [ones(m, 1) X];

pause;

%% ======================= Ex 8: Cost Function and Gradient Descent =============

fprintf('Running gradient descent ...\n');

% Choose some alpha value
alpha = 0.01;
num_iters = 400;

% Init Theta and Run Gradient Descent 
theta = zeros(3, 1);
[theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters);
theta_grad = theta;

% Plot the convergence graph
figure;
plot(1:numel(J_history), J_history, '-b', 'LineWidth', 2);
xlabel('Number of iterations');
ylabel('Cost J');

pause;

%% ======================= Ex 10: Gradient Descent =============

figure;
fprintf('Choose parameter alpha ...\n');

alpha = 0.01; 
theta = zeros(3, 1);
[theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters);
hold on;
plot(1:numel(J_history), J_history, '-b', 'LineWidth', 2)
  
alpha = 0.02; 
theta = zeros(3, 1);
[theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters);  
hold on;
plot(1:numel(J_history), J_history, '-r', 'LineWidth', 2)

alpha = 0.03; 
theta = zeros(3, 1);
[theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters);  
hold on;
plot(1:numel(J_history), J_history, '-g', 'LineWidth', 2)

legend('alpha = 0.01', 'alpha = 0.02', 'alpha = 0.03')

pause;
%% ======================= Ex 11: Normal Equations =============


fprintf('Solving with normal equations...\n');

%% Load Data
data = csvread('ex1data2.txt');
X = data(:, 1:2);
y = data(:, 3);
m = length(y);

% Add intercept term to X
X = [ones(m, 1) X];

% Calculate the parameters from the normal equation
theta = normalEqn(X, y);

% Display normal equation's result
fprintf('Theta computed from the normal equations and gradient descent: \n');
fprintf(' %f \n', theta);
fprintf('\n');

example = [1 1650 3];

example_grad = [1650 3]
example_grad = (example_grad .- mu) ./ sigma;
example_grad = [1, example_grad];
price_normal = example * theta;
price_grad = example_grad * theta_grad;

pause;

fprintf(['Predicted price of a 1650 sq-ft, 3 br house ' ...
         '(using normal equations):\n $%f\n'], price_normal);

fprintf(['Predicted price of a 1650 sq-ft, 3 br house ' ...
         '(using grad equations):\n $%f\n'], price_grad);


