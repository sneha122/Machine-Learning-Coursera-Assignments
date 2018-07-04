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

l = length(theta);
hx = X*theta;
sigm = sigmoid(hx);
identity = ones(m,1);
value = y'*log(sigm)+(identity-y)'*log(identity-sigm);
J1 = (-1/m)*(value); 
J2 = (lambda/(2*m))*(sum(theta(2:l,1).^2));
J = J1+J2;




grad = (1/m)*(X'*(sigm-y));



% =============================================================

end
