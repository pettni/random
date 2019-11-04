function X = Nonlinear_Triangulation(K, C1, R1, C2, R2, C3, R3, x1, x2, x3, X0)
%% Nonlinear_Triangulation
% Refining the poses of the cameras to get a better estimate of the points
% 3D position
% Inputs: 
%     K - size (3 x 3) camera calibration (intrinsics) matrix
%     x1, x2, x3: size (N x 2) matrices
% Outputs: 
%     X - size (N x 3) matrix of refined point 3D locations 

	X = X0;

	for i=1:size(X0, 1)
		X(i, :) = Single_Point_Nonlinear_Triangulation(K, C1, R1, C2, R2, C3, R3, x1(i, :), x2(i, :), x3(i, :), X0(i, :));
	end

end

function X = Single_Point_Nonlinear_Triangulation(K, C1, R1, C2, R2, C3, R3, x1, x2, x3, X0)

	u_x1 = K*R1*(X0'-C1);
	u_x2 = K*R2*(X0'-C2);
	u_x3 = K*R3*(X0'-C3);

	J = [Jacobian_Triangulation(C1, R1, K, X0)' Jacobian_Triangulation(C2, R2, K, X0)' Jacobian_Triangulation(C3, R3, K, X0)']';
	b = [x1 x2 x3]';
	fX = [u_x1(1:2)/u_x1(3); u_x2(1:2)/u_x2(3); u_x3(1:2)/u_x3(3)];

  e = b - fX;
  dx = (J'*J)\(J'*e);

	X = X0 + dx';
end

function J = Jacobian_Triangulation(C, R, K, X)

	f = K(1,1);
	px = K(1,3);
	py = K(2,3);

	u_x = K*R*(X'-C);
	u = u_x(1);
	v = u_x(2);
	w = u_x(3);

	du_dX = [f*R(1,1)+px*R(3,1) f*R(1,2)+px*R(3,2) f*R(1,3)+px*R(3,3)];
	dv_dX = [f*R(2,1)+py*R(3,1) f*R(2,2)+py*R(3,2) f*R(2,3)+py*R(3,3)];
	dw_dX = [R(3,1) R(3,2) R(3,3)];

	J = [w*du_dX-u*dw_dX; w*dv_dX-v*dw_dX]/w^2;
end
