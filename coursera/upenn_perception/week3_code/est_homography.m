function [ H ] = est_homography(video_pts, logo_pts)
% est_homography estimates the homography to transform each of the
% video_pts into the logo_pts
% Inputs:
%     video_pts: a 4x2 matrix of corner points in the video
%     logo_pts: a 4x2 matrix of logo points that correspond to video_pts
% Outputs:
%     H: a 3x3 homography matrix such that logo_pts ~ H*video_pts
% Written for the University of Pennsylvania's Robotics:Perception course

% YOUR CODE HERE

a = [-video_pts -ones(4,1) zeros(4,3) [video_pts ones(4,1)].*repmat(logo_pts(:,1), 1, 3);
		 zeros(4,3) -video_pts -ones(4,1) [video_pts ones(4,1)].*repmat(logo_pts(:,2), 1, 3)];

[U,S,V] = svd(a);
h = V(:, end);

H = mat(h)';

end

