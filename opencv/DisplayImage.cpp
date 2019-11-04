#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;

int main (int argc, char** argv) {

  if (argc != 2) {
    cout << "usage: DisplayImage.out <Image_path>\n" << endl;
    return -1;
  }

  cv::Mat image = cv::imread(argv[1], cv::IMREAD_COLOR);

  if (!image.data) {
    cout << "No image data" << endl;
    return -1;
  }

  // convert to grayscale
  cv::Mat new_image;
  cv::cvtColor(image, new_image, cv::COLOR_BGR2GRAY);

  // convolute with a sharpening filter
  cv::Mat kernel = (cv::Mat_<char>(3,3) << 0, -1, 0, -1, 5, -1, 0, -1, 0);
  cv::filter2D(new_image, new_image, new_image.depth(), kernel);


  ///// display original and new images //////
  cv::namedWindow("Display Image", cv::WINDOW_AUTOSIZE);
  cv::namedWindow("Manipulated Image", cv::WINDOW_AUTOSIZE);
  cv::imshow("Display Image", image);
  cv::imshow("Manipulated Image", new_image);

  // save the new image
  // cv::imwrite("new_image.jpg", manipulated_image);
  
  cv::waitKey(0);
  
  return 0;
}
