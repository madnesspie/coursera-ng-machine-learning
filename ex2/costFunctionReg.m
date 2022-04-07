function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta



h = sigmoid(X * theta);
J = 1 / m * (-y' * log(h) - (1 - y)' * log(1 - h)) + lambda * sum(theta(2:end) .^ 2) / (2 * m);

grad = zeros(size(theta));
for j = 1:length(theta)
    
    p = 0;
    for i = 1:m
        p = p + (h(i) - y(i)) * X(i,j);
    end

    if j == 1
        grad(j) = p / m;
    else
        grad(j) = (p / m) + lambda * theta(j) / m;
    end

end

% =============================================================

end
