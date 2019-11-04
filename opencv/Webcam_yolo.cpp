#include <iostream>

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

#define MODEL "../yolov3-tiny.cfg"
#define WEIGHTS "../yolov3-tiny.weights"
#define CLASSES "../coco.names"

#define INP_WIDTH 416
#define INP_HEIGHT 416

#define CONF_TRESHOLD 0.5
#define NMS_TRESHOLD 0.4

using namespace std;

int main(int argc, char** argv){
 
  cv::VideoCapture cap(0); 
    
  if(!cap.isOpened()){
    cout << "Error opening video stream or file" << endl;
    return -1;
  }

  cv::namedWindow("Webcam", cv::WINDOW_AUTOSIZE);
  cv::Mat frame;

  // read net
  cv::dnn::Net net = cv::dnn::readNetFromDarknet(MODEL, WEIGHTS);
  net.setPreferableBackend(cv::dnn::DNN_BACKEND_DEFAULT);
  net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

  // read classes
  vector<string> classes;
  ifstream ifs(CLASSES);
  string line;
  while (getline(ifs, line))
      classes.push_back(line);

  while(1){

    // storage
    cv::Mat blob;
    vector<cv::Mat> outs;
    vector<int> classIds;
    vector<float> confidences;
    vector<cv::Rect> boxes;

    // read frame
    cap >> frame;
    if (frame.empty())
      break;

    // preprocess frame
    cv::dnn::blobFromImage(frame, blob, 1.0/255.0, cv::Size(416, 416), cv::Scalar(0,0,0), true, false, CV_32F);

    // net forward pass
    net.setInput(blob);
    net.forward(outs, net.getUnconnectedOutLayersNames());

    // postprocess
    for (size_t i = 0; i < outs.size(); ++i) {
      // Network produces output blob with a shape NxC where N is a number of
      // detected objects and C is a number of classes + 4 where the first 4
      // numbers are [center_x, center_y, width, height]
      float* data = (float*)outs[i].data;
      for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols) {

        cv::Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
        cv::Point classIdPoint;
        double confidence;
        cv::minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
        if (confidence > CONF_TRESHOLD) {
            int centerX = (int)(data[0] * frame.cols);
            int centerY = (int)(data[1] * frame.rows);
            int width = (int)(data[2] * frame.cols);
            int height = (int)(data[3] * frame.rows);
            int left = centerX - width / 2;
            int top = centerY - height / 2;

            classIds.push_back(classIdPoint.x);
            confidences.push_back((float)confidence);
            boxes.push_back(cv::Rect(left, top, width, height));
        }
      }
    }

    // non-maximum suppression
    vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, CONF_TRESHOLD, NMS_TRESHOLD, indices);

    // draw boxes
    for (size_t i = 0; i < indices.size(); ++i) {
        int idx = indices[i];
        cv::Rect box = boxes[idx];

        // draw rectangle
        cv::rectangle(frame, cv::Point(box.x, box.y), 
                      cv::Point(box.x+box.width, box.y+box.height), 
                      cv::Scalar(255,0,0));  
        string label = classes[classIds[idx]] + ": " + cv::format("%.2f", confidences[idx]);        
        cv::putText(frame, label, cv::Point(box.x, box.y), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar());    
    }    

    cv::imshow("Webcam", frame);
 
    char c = (char) cv::waitKey(25);
    if(c==27)
      break;
  }
  
  cap.release();
  cv::destroyAllWindows();
     
  return 0;
}
