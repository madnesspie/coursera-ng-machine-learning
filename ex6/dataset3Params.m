function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

params = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
tries = length(params);
paramPairs = zeros(tries * 2, 2);
errors = zeros(tries * 2, 1);
counter = 0;

for i = 1:tries
    for j = 1:tries
        counter = counter + 1
        
        C = params(i);
        sigma = params(j);
        model = svmTrain(X, y, C, @(x1, x2)gaussianKernel(x1, x2, sigma));
        predictions = svmPredict(model, Xval);

        paramPairs(counter, 1) = C;
        paramPairs(counter, 2) = sigma;
        errors(counter) = mean(double(predictions ~= yval));
    end
end

[minErr, idx] = min(errors);
fprintf('minErr = %f \n', minErr);
C = paramPairs(idx, 1);
sigma = paramPairs(idx, 2);

% =========================================================================

end
