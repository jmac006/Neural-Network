function [activation,hiddenOutput,output] = forwardPropagation(X,W1,W2)
	activation = X*W1; %use for activation function
    hiddenOutput = sigmoid(activation);
    hiddenOutput = [ones(size(hiddenOutput,1),1) hiddenOutput];
    output = sigmoid(hiddenOutput*W2);