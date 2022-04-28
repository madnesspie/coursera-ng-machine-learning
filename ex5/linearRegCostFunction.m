function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

m = length(y); % number of training examples
h = X * theta;
regterm = lambda * sum(theta(2:end,:) .^ 2) / (2 * m);
J = sum((h - y) .^ 2) / (2 * m) + regterm;
grad = X' * (h - y) / m + (lambda * [0; theta(2:end,:)] / m);

end
