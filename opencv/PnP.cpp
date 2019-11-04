#include <iostream>
#include <opencv2/opencv.hpp>
#include "opencv2/xfeatures2d/nonfree.hpp"

using namespace std;

int main(int argc, char const *argv[])
{

  cv::Mat image1 = cv::imread("../kitchen1.png", cv::IMREAD_COLOR);
  cv::Mat image2 = cv::imread("../kitchen2.png", cv::IMREAD_COLOR);

  cv::Mat image1_gray, image2_gray;
  cv::cvtColor(image1, image1_gray, cv::COLOR_BGR2GRAY);
  cv::cvtColor(image2, image2_gray, cv::COLOR_BGR2GRAY);

  if (!image1.data || !image2.data) {
    cout << "No image data" << endl;
    return -1;
  }

  // Detect features
	cv::Ptr<cv::Feature2D> detector = cv::xfeatures2d::SiftFeatureDetector::create();
	std::vector<cv::KeyPoint> keypoints1, keypoints2;
	detector->detect(image1_gray, keypoints1);
	detector->detect(image2_gray, keypoints2);
	
	// Compute descriptors
	cv::Ptr<cv::DescriptorExtractor> extractor = cv::xfeatures2d::SiftDescriptorExtractor::create();
  cv::Mat descriptors1, descriptors2;
  extractor->compute(image1_gray, keypoints1, descriptors1);
  extractor->compute(image2_gray, keypoints2, descriptors2);

  // Match descriptors
  cv::BFMatcher matcher;
  std::vector<cv::DMatch> matches;
  matcher.match(descriptors1, descriptors2, matches, cv::Mat());

  // Extract best matches
  std::sort(matches.begin(), matches.end());
  std::vector<cv::Point2f> points1, points2;
  for (cv::DMatch match : matches) {
  	points1.push_back(keypoints1[match.queryIdx].pt);
  	points2.push_back(keypoints2[match.trainIdx].pt);
  }

  // RANSAC for homography
  cv::Mat mask;
  const double maxReportjTreshold = 3.0;
  cv::Mat h = cv::findHomography(points1, points2, cv::RANSAC, maxReportjTreshold, mask);

  cout << mask.size() << endl;

  std::vector<cv::DMatch> inliers;
  for (int i=0; i!=matches.size(); ++i) {
  	if (mask.at<int>(i, 0) == 1) {
  		inliers.push_back(matches[i]);
  	}
  }

  cv::Mat img_matches;
  cv::drawMatches(image1, keypoints1, image2, keypoints2,
                  inliers, img_matches, cv::Scalar::all(-1), cv::Scalar::all(-1),
               		vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

  cv::namedWindow("Test", cv::WINDOW_AUTOSIZE);
  cv::imshow("Test", img_matches);
  cv::waitKey(0);

	return 0;
}