% Name: Justin Mac
% SID: 861086907
% Date: 5/10/17
% CS171 Problem Set 3

function s = sigmoid(x)
	%basic fucntion that uses the sigmoid function as an activation function
	s = 1 ./ (1 + exp(-x));