% Name: Justin Mac
% SID: 861086907
% Date: 5/10/17
% CS171 Problem Set 3: Neural Network with 2 layers of weights and 1 hidden
% layer by varying the number of hidden units and lambda
function [W1,W2] = trainneuralnet(X,Y,nhid,lambda)
	% Function returns weights for the first layer and the weights for the second layer
	% nhid = the number of hidden 

	%add a column of 1's to the array X
	X = [ones(size(X,1),1) X]

	
	[m,n] = size(X);
	W1 = randn(3,nhid); %begin with random weights
	W2 = randn(nhid+1,1);
	W1 = 2.*W1-1;
	W2 = 2.*W2-1;
	gradient1 = 1;
	gradient2 = 1;

	w = zeros(nhid,1);

	for iteration = 1:500000 %max iteration should be 500,000 iterations
		if max(max(max(abs(gradient1)))) < 0.0001
			if max(max(max(abs(gradient2)))) < 0.0001
				break %all of the elements of all gradients should be within 0.0001
			end
		end
		learningRate = 0.1; %step size
		gradient1 = 0;
		gradient2 = 0;
		for i = 1:m
			[activate,hiddenOutput,out] = forwardPropagation(X,W1,W2,i); %Forward propagation
			[delta1,delta2] = backwardPropagation(Y,W2,activate,out,i); %Backward propagation
			%calculate the sum
			sumW2 = delta2.*hiddenOutput';
			%replicate matrix into a 1 x # hidden units to duplicate as bipartite graph
			%multiply by replicated matrix of 3 x 1 block
			sumW1 = repmat(X(i,:)',1,nhid).*repmat(delta1,3,1); 
			gradient1 = gradient1+sumW1;
			gradient2 = gradient2+sumW2;
			
		end

		gradient1 = gradient1+2.*lambda.*W1; %batch update gradient equation (from piazza)
		gradient2 = gradient2+2.*lambda.*W2; %using hadamard product
		
		W1 = W1-learningRate.*gradient1; %update the new weights
		W2 = W2-learningRate.*gradient2;
		iteration = iteration + 1;

		if mod(iteration,1000) == 0
			iteration
			gridX = getgridpts(X,20);
			newGridX = [ones(size(gridX,1),1) gridX]; %add a column of 1's to gridX
			Ha = newGridX * W1;
			Hz = 1./(1+exp(-Ha));
			Hz = [ones(size(Hz,1),1) Hz];
			Ya = Hz*W2;
			Yz = 1./(1+exp(-Ya));
			gridY = Yz;
			plotdecision(X,Y,gridX,gridY);
			drawnow
		end
	end