function R = part2(Train500, Train1000, Test, provideIdx, missIdx)
	Train = [Train500;Train1000];
	% X is sample * feature (N * K)
	X = Test;
	% Get mean for feature (1 * K) from Train
	Mean_Train = mean(Train);
	% 0 mean by minus mean
	%Train = Train .- Mean_Train;
    Train = bsxfun(@minus,Train, Mean_Train);
    

	% get V from 1500 training data (5903*5903)
	[U,S,V] = svd(Train);

	% 3172*50
	V1 = V(provideIdx,1:50);
	V2 = V(missIdx,1:50);
	
	% for i = 1 to N get a_i and put result back in to R 1000*(3172+2769)
	% A = 50*N
	A = pinv(V1)*X';

	% pinv
	R = V2*A;

	% Add Back Mean
	%R=R'.+Mean_Train(missIdx);
    R=bsxfun(@plus, R',Mean_Train(missIdx));

	%R = roundn(R,-11);
	
	csvwrite('prediction.csv', R);
end