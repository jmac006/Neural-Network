% Name: Justin Mac
% SID: 861086907
% Date: 5/10/17
% CS171 Problem Set 3

function runq1

toy = load('toy.data','-ascii');
X = toy(:,1:end-1);
Y = toy(:,end);

fnum = 1;

for lambda = [0.001 0.0001 0.00001]
%for lambda = [0.00025] %used for testing to match with outputs on problem set handout
    for nhidden = [1 5 15] %number of hidden units   
		subplot(3,3,fnum);
		[W1,W2] = trainneuralnet(X,Y,nhidden,lambda)
		gridX = getgridpts(X,20); 
		newGridX = [ones(size(gridX,1),1) gridX]; %add a column of 1's to gridX
		[hiddenActivation,hiddenZ,out] = forwardPropagation(newGridX,W1,W2);
		gridY = out;
		plotdecision(X,Y,gridX,gridY);
		title(['nhidden = ' num2str(nhidden) ', lambda = ', num2str(lambda)]);
		hold off;
		fnum = fnum+1;
	end;
end;
set(gcf,'paperorientation','landscape');
set(gcf,'paperposition',[0.25 0.25 10.25 8.25]); %dimensions for paper
saveas(gcf,'q1out.pdf');


