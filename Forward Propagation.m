function [a,z,f] = forward_propagation(X,W1,W2)
	a = X(i,:)*W1;
    z = 1./(1+exp(-a));
    z = [ones(size(z,1),1) z];
    f = 1/(1+exp(-z*W2));