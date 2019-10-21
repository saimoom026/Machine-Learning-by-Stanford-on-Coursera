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

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
             % a1=x, a2=sig(th1*a1), a3=sig(th2*a2), a3=ht

% Add ones to the X data matrix
a1=[ones(m, 1) X];
a2=[ones(m, 1) sigmoid(a1*Theta1')];
a3= sigmoid(a2*Theta2');
h=a3;

grad=0;
ybin=0;
for k=1:num_labels
y_t=y==k;
if k==1             % binary representation of Y
ybin=y_t;
else
ybin=[ybin y_t];     
end
temp=sum(y_t.*log(h(:,k))+(1-y_t).*log(1-h(:,k)));
J=J+(-1/m)*temp;
end

% regularization
Theta1_t=Theta1;
Theta2_t=Theta2;
Theta1_t(:,1)=[];
Theta2_t(:,1)=[];
for j=1:hidden_layer_size 
sm1=sum(Theta1_t(j,:).^2);
sm2=sum(Theta2_t(:,j).^2);
J=J+(lambda/m/2)*(sm1+sm2); 
end

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
del_ij1=0;
del_ij2=0;

% for i=1:m
delta3=a3-ybin;
delta2=delta3*Theta2.*a2.*(1-a2);
delta2=delta2(:,2:end);

del_ij2=del_ij2+delta3'*a2;
del_ij1=del_ij1+delta2'*a1;
% end

Theta1_grad = (1/m)*del_ij1;
Theta2_grad = (1/m)*del_ij2;


% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
% set first column to 0
Theta1(:,1)=0;
Theta2(:,1)=0;
% scale it (lamda/m)
Theta1=(lambda/m).*Theta1+Theta1_grad ;
Theta2=(lambda/m).*Theta2+Theta2_grad ;

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1(:) ; Theta2(:)];


end
