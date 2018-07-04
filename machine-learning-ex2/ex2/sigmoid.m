function g = sigmoid(z)
%SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).


identity_matrix = ones(size(z));
exponent_matrix = exp(-z);
simple_matrix = identity_matrix + exponent_matrix;
g = simple_matrix.^-1;



% =============================================================

end
