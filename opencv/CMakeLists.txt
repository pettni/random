cmake_minimum_required(VERSION 2.8)
project( DisplayImage )

find_package( OpenCV 4 REQUIRED)
set(OPENCV_ENABLE_NONFREE ON)

include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable( DisplayImage DisplayImage.cpp )
target_link_libraries( DisplayImage ${OpenCV_LIBS} )

add_executable( PnP PnP.cpp )
target_link_libraries( PnP ${OpenCV_LIBS} )

add_executable( Webcam_tracking Webcam_tracking.cpp )
target_link_libraries( Webcam_tracking ${OpenCV_LIBS} )

add_executable( Webcam_yolo Webcam_yolo.cpp )
target_link_libraries( Webcam_yolo ${OpenCV_LIBS} )
