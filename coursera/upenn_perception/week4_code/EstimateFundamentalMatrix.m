function F = EstimateFundamentalMatrix(x1, x2)
%% EstimateFundamentalMatrix
% Estimate the fundamental matrix from two image point correspondences 
% Inputs:
%     x1 - size (N x 2) matrix of points in image 1
%     x2 - size (N x 2) matrix of points in image 2, each row corresponding
%       to x1
% Output:
%    F - size (3 x 3) fundamental matrix with rank 2

	N = size(x1, 1);

	A = zeros(N, 9);
	for i=1:N
		A(i,:) = kron([x1(i,:) 1], [x2(i, :) 1]);
	end

	% solve least-squares
	[u,s,v] = svd(A);
	F = mat(v(:, end));

	% cleanup
	[u,s,v] = svd(F);
	s(3,3) = 0;
	F = u * s * v';

	F = F/norm(F);