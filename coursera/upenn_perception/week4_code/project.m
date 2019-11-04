%% project: project from 3D point to image plane
function [x] = project(K, C, R, X)
	x = K*R*[X'-repmat(C, size(X,1))];
	x = x(1:2, :)./repmat(x(3,:), 2, 1);
	x = x';
