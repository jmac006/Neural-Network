% Name: Justin Mac
% SID: 861086907
% Date: 5/10/17
% CS171 Problem Set 3

function [activation,hiddenOutput,output] = forwardPropagation(X,W1,W2)
	%Input: matrix X, and weights W1 and W2
	%Returns the activation function based on W1, the hidden output by 
	%applying the sigmoid function, and output that is used for the next hidden layer/neuron.
	activation = X*W1; %use for activation function
    hiddenOutput = sigmoid(activation);
    %add 1 to hiddenOutput's first column so matrix dimensions agree for multiplication
    hiddenOutput = [ones(size(hiddenOutput,1),1) hiddenOutput];
    output = sigmoid(hiddenOutput*W2); %1/1+e^(-hiddenOutput*W2)