function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

% Calcuate J (cost function)
total_cost = 0;
for i = 1:m
	part_1 = log(sigmoid(dot(transpose(theta) , X(i,:))));
	part_2 = log(1 - sigmoid(dot(transpose(theta) , X(i,:))));
	total_cost = total_cost + (-y(i,1) * part_1 - (1 - y(i,1)) * part_2 ) ;
end
% Regularization part
n = length(theta);
total_theta =0;
for i=2:n
	total_theta = total_theta + theta(i,1) * theta(i,1);
end
% Conclusion for J
J = (1/m) * total_cost + total_theta*lambda/(2*m);

% Gradient Part
for j = 1:(length(theta))
	total_gradient = 0;
	for i = 1:m
		h=sigmoid(dot(transpose(theta) , X(i,:)));
		total_gradient = total_gradient + (h - y(i,1)) * X(i,j);
	endfor
	if j==1
		grad(j,1) = (1/m) * total_gradient;
	else	
		grad(j,1) = (1/m) * total_gradient + (lambda/m)* theta(j,1);
	end
end

% =============================================================

end
