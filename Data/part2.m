function R = part2(Train500, Train1000, Test, provideIdx, missIdx, N)
    %N = 55;
	Train = [Train500;Train1000];
    
	% X is sample * feature (rows * K) (1000* 3172)
	X = Test;
    %Mean_X = mean(X);
    %X = bsxfun(@minus,X, Mean_X);
	% Get mean for feature (1 * K) from Train
	Mean_Train = mean(Train);
    S_Train = std(Train);
    
	% 0 mean by minus mean
	%Train = Train .- Mean_Train;
    Train = bsxfun(@minus,Train, Mean_Train);
    X = bsxfun(@minus,X, Mean_Train(:,provideIdx));
    
    Train = bsxfun(@rdivide,Train, S_Train);
    X = bsxfun(@rdivide,X, S_Train(:,provideIdx));
    
	% get V from 1500 training data (5903*5903)
	[U,S,V] = svd(Train);

	% 3172*50 if N = 50
	V1 = V(provideIdx,1:N);
    
    % 2796 * 50
	V2 = V(missIdx,1:N);
	
	% for i = 1 to N get a_i and put result back in to R 1000*(3172+2769)
	% A = 50*N  % pinv   X 1000* 3172   pinv(V1) 50* 3172
	A = pinv(V1)*X';

	% 2796* 50 * 50 * N
	R = V2*A;

	% Add Back Mean
	%R=R'.+Mean_Train(missIdx);
    R = R';
    R=bsxfun(@times, R,S_Train(missIdx));
    R=bsxfun(@plus, R,Mean_Train(missIdx));
    
    % unmean X
    % 55=>57.1 60=>57.6 63=>57.5, 65=>57.7
    % 70=>57.4 64=>57.7 66 => 57.8 67=> 57.8 68=> 57.7
    % 67/7 => 57.8  67/6=> 57.8
    % mean 0 X
    % 67=> 57.5 60=> 57.3 70 => 57.0 65=> 57.4
    % mean 0 X with Train
    % 66-> 57.8 68-> 57.7 6 - 57.7
    % no mean at all
	% 66-> 58 67-> 57.8 65-> 57.7
    % add std
    % 66-> 58.4 67 -> 58.5 68-> 58.4
    dlmwrite('prediction.csv',R,'delimiter', ',', 'precision',6)
	%csvwrite('prediction.csv', R);
end