#include <iostream>

#include "opencv2/opencv.hpp"
#include <opencv2/tracking/tracker.hpp>
 
using namespace std;
using namespace cv;
 
int main(){
 
   VideoCapture cap(0); 
    
  if(!cap.isOpened()){
    cout << "Error opening video stream or file" << endl;
    return -1;
  }

  cv::namedWindow("Webcam", cv::WINDOW_AUTOSIZE);  
  
  Mat frame;
  Rect2d bbox;

  cv::TrackerMIL::Params params;
  cv::Ptr<cv::Tracker> tracker = cv::TrackerMIL::create(params);

  cap >> frame;
  bbox = selectROI(frame, false);

  cout << "Selected bbox " << bbox << ", initializing tracker" << endl;

  tracker->init(frame, bbox);
     
  while(1){
    cap >> frame;
    if (frame.empty())
      break;

    bool ok = tracker->update(frame, bbox);

    if (ok) 
      cv::rectangle(frame, bbox, cv::Scalar(255,0,0), 2, 1);
    else
      cv::putText(frame, "Trakcing failed", cv::Point(100,80), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255,0,0));
 
    imshow("Webcam", frame);
 
    char c = (char)waitKey(25);
    if(c==27)
      break;
  }
  
  cap.release();
  destroyAllWindows();
     
  return 0;
}