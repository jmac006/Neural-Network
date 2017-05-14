function [activation,hiddenOutput,output] = forwardPropagation(X,W1,W2,i)
	activation = X(i,:)*W1; %use for activation function
    hiddenOutput = sigmoid(activation);
    hiddenOutput = [ones(size(hiddenOutput,1),1) hiddenOutput];
    output = 1/(1+exp(-hiddenOutput*W2));