function [delta2,delta1] = backwardPropagation(Y,W2,a,f,i)
    delta1 = f - Y(i,:);
	temp = delta1*W2(2:end,:)';
	delta2  = exp(-a)./((1+exp(-a)).^2).*temp;