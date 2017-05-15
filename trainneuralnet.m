% Name: Justin Mac
% SID: 861086907
% Date: 5/10/17
% CS171 Problem Set 3: Neural Network with 2 layers of weights and 1 hidden
% layer by varying the number of hidden units and lambda

function [W1,W2] = trainneuralnet(X,Y,nhid,lambda)
	% Function returns weights for the first layer and the weights for the second layer
	% nhid = the number of hidden 

	origX = X; %needed to use drawnow
	%add a column of 1's to the array X
	X = [ones(size(X,1),1) X];
	
	[m,n] = size(X);
	W1 = randn(3,nhid); %begin with random weights
	W2 = randn(nhid+1,1);
	W1 = 2.*W1-1;
	W2 = 2.*W2-1;
	gradient1 = 1;
	gradient2 = 1;

	%hiddenWeights = zeros(nhid,1);

	%iterations should take 100,000-500,000 iterations to converge
	for iteration = 1:500000 %max iteration should be 500,000 iterations
		if max(max(max(abs(gradient1)))) < 0.0001
			if max(max(max(abs(gradient2)))) < 0.0001
				%gradient1
				%gradient2
				break %all of the elements of all gradients should be within 0.0001
			end
		end
		learningRate = 0.1; %step size
		gradient1 = 0;
		gradient2 = 0;
		for i = 1:m
			%Forward propagation, need to pass in index i
			[activate,hiddenOutput,out] = forwardPropagation(X(i,:),W1,W2);
			%Backward propagation, need to pass in index i
			[delta1,delta2] = backwardPropagation(out,Y(i,:),W2,activate); 
			%find the sum of all weights
			sumW2 = delta2.*hiddenOutput';
			%replicate matrix into a 1 x # hidden units to duplicate as bipartite graph
			%multiply by replicated matrix of 3 x 1 block
			sumW1 = repmat(X(i,:)',1,nhid).*repmat(delta1,3,1);
			%add new sum of weights to gradients
			gradient1 = gradient1+sumW1;
			gradient2 = gradient2+sumW2;
			
		end

		gradient1 = gradient1+2.*lambda.*W1; %batch update gradient equation (from piazza)
		gradient2 = gradient2+2.*lambda.*W2;
		
		W1 = W1-learningRate.*gradient1; %update the new weights
		W2 = W2-learningRate.*gradient2;
		
		if mod(iteration,1000) == 0 %update the plot every 1000 iterations
			iteration
			gridX = getgridpts(origX,20); 
			newGridX = [ones(size(gridX,1),1) gridX]; %add a column of 1's to gridX
			gridY = zeros(size(gridX, 1), 1);
			[hiddenActivation,hiddenZ,out] = forwardPropagation(newGridX,W1,W2);
			gridY = out;
			gridY(gridY > 0.5) = 1;
			gridY(gridY <= 0.5) = 0;
			plotdecision(X(:, 2:end), Y, gridX, gridY)
			title(['nhidden = ' num2str(nhid) ', lambda = ', num2str(lambda)]);
			drawnow
		end

		iteration = iteration + 1;
	end