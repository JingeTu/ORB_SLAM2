/*****************************************************************************
*   Cartoonifier_Desktop.cpp, for Desktop.
*   Converts a real-life camera stream to look like a cartoon.
*   This file is for a desktop executable, but the cartoonifier can also be used in an Android / iOS project.
******************************************************************************
*   by Shervin Emami, 5th Dec 2012 (shervin.emami@gmail.com)
*   http://www.shervinemami.info/
******************************************************************************
*   Ch1 of the book "Mastering OpenCV with Practical Computer Vision Projects"
*   Copyright Packt Publishing 2012.
*   http://www.packtpub.com/cool-projects-with-opencv/book
*****************************************************************************/


// Try to set the camera resolution. Note that this only works for some cameras on
// some computers and only for some drivers, so don't rely on it to work!
const int DESIRED_CAMERA_WIDTH = 1024;
const int DESIRED_CAMERA_HEIGHT = 768;

// Set to true if you want to see line drawings instead of paintings.
bool m_sketchMode = false;
// Set to true if you want to change the skin color of the character to an alien color.
bool m_alienMode = false;
// Set to true if you want an evil "bad" character instead of a "good" character.
bool m_evilMode = false;
// Set to true if you want to see many windows created, showing various debug info. Set to 0 otherwise.
bool m_debugMode = false;
// Set to true if you want to save the next frame.
bool m_saveCurrentFrame = false;


#include <stdio.h>
#include <stdlib.h>
#include <ctime>
#include <chrono>

#include <System.h>

// Include OpenCV's C++ Interface
#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;

// Get access to the webcam.
void initWebcam(VideoCapture &videoCapture, int cameraNumber)
{
    // Get access to the default camera.
    try {   // Surround the OpenCV call by a try/catch block so we can give a useful error message!
        videoCapture.open(cameraNumber);
    } catch (cv::Exception &e) {}
    if ( !videoCapture.isOpened() ) {
        cerr << "ERROR: Could not access the camera!" << endl;
        exit(1);
    }
    cout << "Loaded camera " << cameraNumber << "." << endl;
}


int main(int argc, char *argv[])
{

    // Allow the user to specify a camera number, since not all computers will be the same camera number.
    int cameraNumber = 0;   // Change this if you want to use a different camera device.
    // if (argc > 1) {
    //     cameraNumber = atoi(argv[1]);
    // }
    
    // Retrieve paths to images
    vector<double> vTimestamps;


    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    ORB_SLAM2::System SLAM(argv[1],argv[2],ORB_SLAM2::System::MONOCULAR,true);

    // Get access to the camera.
    VideoCapture camera;
    initWebcam(camera, cameraNumber);

    // Try to set the camera resolution. Note that this only works for some cameras on
    // some computers and only for some drivers, so don't rely on it to work!
    camera.set(CV_CAP_PROP_FRAME_WIDTH, DESIRED_CAMERA_WIDTH);
    camera.set(CV_CAP_PROP_FRAME_HEIGHT, DESIRED_CAMERA_HEIGHT);

    std::chrono::steady_clock::time_point t1;

    // Run forever, until the user hits Escape to "break" out of this loop.
    while (true) {

        // Grab the next camera frame. Note that you can't modify camera frames.
        Mat cameraFrame;
        camera >> cameraFrame;
        if( cameraFrame.empty() ) {
            cerr << "ERROR: Couldn't grab the next camera frame." << endl;
            exit(1);
        }

        // std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
        // double tframe =  std::chrono::duration_cast<std::chrono::duration<double> >(t1).count();
        SLAM.TrackMonocular(cameraFrame,0.0);

        // waitKey(20);
        // imshow(windowName, cameraFrame);

    }//end while

    return EXIT_SUCCESS;
}
