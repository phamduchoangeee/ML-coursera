function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

size_X = size(X);    % 12x2    
size_y = size(y);	 % 12x1
size_theta = size(theta); % 2x1

% part 1: calculate Error Part
error_part = 0;
h = X*theta; % 12x1
error_part = sum((h - y).^2);

% part 2: calculate regularized Part
regularized_part = 0;
theta_reg = theta;
theta_reg(1,1) = 0;
regularized_part = sum(theta_reg.^2);

J = 1/(2*m)* error_part + lambda/(2*m)*regularized_part;

grad_main_part = 1/m * transpose(X) * (h-y);
grad_reg_part  = lambda/m * theta_reg;
grad = grad_main_part + grad_reg_part;







% =========================================================================

grad = grad(:);

end
