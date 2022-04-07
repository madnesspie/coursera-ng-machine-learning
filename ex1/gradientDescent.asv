function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);


for iter = 1:num_iters

    delta = zeros(size(theta));
    for j = 1:length(theta)
   
        p = 0;
        for i = 1:length(X)
            h = theta' * X(i, :)';
            s = (h - y(i)) * X(i,j);
            p = p + s;
        end    
        
        delta(j,1) = p / m;
    end

    % Simultaneously update theta
    theta = theta - alpha * delta;
    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
