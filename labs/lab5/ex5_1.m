%% Initialization
clear ; close all; clc

%% =============== Part 1-2: Loading and Visualizing Data ================

fprintf('Loading and Visualizing Data ...\n')

% Load from ex5data1: 
% You will have X, y in your environment
load('ex5data1.mat');

% Plot training data
plotData(X, y);

pause;

%% ==================== Part 3-4: Training Linear SVM ====================

% Load from ex5data1: 
% You will have X, y in your environment
load('ex5data1.mat');

fprintf('\nTraining Linear SVM ...\n')

C = 1;
model = svmTrain(X, y, C, @linearKernel, 1e-3, 20);
visualizeBoundaryLinear(X, y, model);

pause;

%% =============== Part 5: Implementing Gaussian Kernel ===============
fprintf('\nEvaluating the Gaussian Kernel ...\n')

x1 = [1 2 1]; x2 = [0 4 -1]; sigma = 2;
sim = gaussianKernel(x1, x2, sigma);

fprintf(['Gaussian Kernel between x1 = [1; 2; 1], x2 = [0; 4; -1], sigma = %f :' ...
         '\n\t%f\n(for sigma = 2, this value should be about 0.324652)\n'], sigma, sim);

pause;

%% =============== Part 6: Visualizing Dataset 2 ================
fprintf('Loading and Visualizing Data ...\n')

% Load from ex5data2: 
% You will have X, y in your environment
load('ex5data2.mat');

% Plot training data
plotData(X, y);

pause;

%% ========== Part 7-9: Training SVM with RBF Kernel (Dataset 2) ==========
fprintf('\nTraining SVM with RBF Kernel (this may take 1 to 2 minutes) ...\n');

% Load from ex5data2: 
% You will have X, y in your environment
load('ex5data2.mat');

% SVM Parameters
C = 1; sigma = 0.1;

model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma)); 
visualizeBoundary(X, y, model);

pause;

%% =============== Part 10: Visualizing Dataset 3 ================

fprintf('Loading and Visualizing Data ...\n')

% Load from ex5data3: 
% You will have X, y in your environment
load('ex5data3.mat');

% Plot training data
plotData(X, y);

pause;

%% ========== Part 11: Training SVM with RBF Kernel (Dataset 3) ==========

% Load from ex5data3: 
% You will have X, y in your environment
load('ex5data3.mat');

% Try different SVM Parameters here
[C, sigma] = dataset3Params(X, y, Xval, yval);

%% ========== Part 12: Training SVM with RBF Kernel (Dataset 3) ==========

% Train the SVM
model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
visualizeBoundary(X, y, model);

fprintf('Program paused. Press enter to continue.\n');
pause;

