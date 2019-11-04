function [C, R] = LinearPnP(X, x, K)
%% LinearPnP
% Getting pose from 2D-3D correspondences
% Inputs:
%     X - size (N x 3) matrix of 3D points
%     x - size (N x 2) matrix of 2D points whose rows correspond with X
%     K - size (3 x 3) camera calibration (intrinsics) matrix
% Outputs:
%     C - size (3 x 1) pose transation
%     R - size (3 x 1) pose rotation
%
% IMPORTANT NOTE: While theoretically you can use the x directly when solving
% for the P = [R t] matrix then use the K matrix to correct the error, this is
% more numeically unstable, and thus it is better to calibrate the x values
% before the computation of P then extract R and t directly


	N = size(X, 1);

	A = zeros(3*N, 12);

	for i=1:N
		A(1+3*i:3+3*i, :) = kron([X(i,:) 1], Vec2Skew(inv(K)*[x(i,:)'; 1]));
	end

	% solve least-squares
	[u,s,v] = svd(A);
	P = reshape(v(:, end), 3, 4);

	% extract P
	% KmP = inv(K)*P;
	KmP = P;

	% clean up rotation
	Rc = KmP(:, 1:3);
	t = KmP(:, 4);

	[u,s,v] = svd(Rc);

	if det(u*v') > 0
		R = u*v';
		tc = t/s(1,1);
	else
		R = -u*v';
		tc = -t/s(1,1);
	end

	C = -R'*tc; 