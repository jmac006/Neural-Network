% Name: Justin Mac
% SID: 861086907
% Date: 5/10/17
% CS171 Problem Set 3

function [delta1,delta2] = backwardPropagation(output,Y,W2,activation)
	%Backward propagation takes the predicted output and updates the new value of weights
    delta2 = output - Y;
	classfier = delta2*W2(2:end,:)'; %take all the weights transpose and multiply it by delta
	%derivative of sigmoid in respect to x (or the net input)
	delta1  = exp(-activation)./((1+exp(-activation)).^2); 
	delta1 = delta1.*classfier; %using equation found in lecture; uses hadamard product