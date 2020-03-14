function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


X = [ones(m,1),X];
total = 0;
% Theta1_size : 25x401
% Theta2_size : 10x26

% recode y
y_recode = zeros(num_labels, m);
for i= 1:m
	y_recode(y(i,1), i) = 1;
end	

% calculus hidden layer
z_2 = X * transpose(Theta1);        % size: 5000x25
hidden_layer = sigmoid(z_2);		
% add ones to hidden layer
hidden_layer = [ones(m,1),hidden_layer];    % size: 5000x26
% calculus output layer
z_3 = hidden_layer * transpose(Theta2);     % size: 5000x10
output_layer = sigmoid(z_3);

for i = 1:m
	part_1 = (-1)*sum(transpose(y_recode(:,i)).*log(output_layer(i,:)));
	part_2 = (-1)*sum(transpose(1 - y_recode(:,i)).*log(1 - output_layer(i,:)));
	total = total + part_1 + part_2;
end
% cost
J = total/m;
regularized_Theta_1 = Theta1(:,2:end);
regularized_Theta_2 = Theta2(:,2:end);
regularized_part = lambda/(2*m) * (sum(sum(regularized_Theta_1.*regularized_Theta_1)) + sum(sum(regularized_Theta_2.*regularized_Theta_2)));

J = J + regularized_part;

% gradient
delta_3 = output_layer - transpose(y_recode);    % size: 5000x10
delta_2 = (delta_3 * Theta2) .* sigmoidGradient([ones(m,1),z_2]);  % size (temp) : 5000x26
delta_2 = delta_2(:,2:end); % size: 5000x25

% delta_3: 5000x10        hidden_layer: 5000x26
triangle_2 = transpose(delta_3) * hidden_layer; % size: 10 x 26
% delta_2: 5000x25        X: 5000x401
triangle_1 = transpose(delta_2) * X; % size: 25x401

Theta1_reg = [zeros(size(Theta1,1),1) , Theta1(:,2:end)];
Theta2_reg = [zeros(size(Theta2,1),1) , Theta2(:,2:end)];

Theta1_grad = triangle_1/m + lambda/m * (Theta1_reg);
Theta2_grad = triangle_2/m + lambda/m * (Theta2_reg);



% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
