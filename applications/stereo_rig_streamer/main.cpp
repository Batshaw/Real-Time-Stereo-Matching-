/* This is sample from the OpenCV book. The copyright notice is below */

/* *************** License:**************************
   Oct. 3, 2008
   Right to use this code in any way you want without warranty, support or any guarantee of it working.

   BOOK: It would be nice if you cited it:
   Learning OpenCV: Computer Vision with the OpenCV Library
     by Gary Bradski and Adrian Kaehler
     Published by O'Reilly Media, October 3, 2008

   AVAILABLE AT:
     http://www.amazon.com/Learning-OpenCV-Computer-Vision-Library/dp/0596516134
     Or: http://oreilly.com/catalog/9780596516130/
     ISBN-10: 0596516134 or: ISBN-13: 978-0596516130

   OPENCV WEBSITES:
     Homepage:      http://opencv.org
     Online docs:   http://docs.opencv.org
     Q&A forum:     http://answers.opencv.org
     GitHub:        https://github.com/opencv/opencv/
   ************************************************** */

#include "opencv2/calib3d.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include <BSystem.h>
#include <BCamera.h>

#include <vector>
#include <string>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <iterator>
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <boost/program_options.hpp>
#include <boost/algorithm/string.hpp>

using namespace cv;
using namespace std;


std::shared_ptr<baumer::BSystem> baumer_system  = nullptr;



int main(int argc, char** argv) {

  baumer_system = std::make_shared<baumer::BSystem>();

  baumer_system->init();

  uint32_t const num_detected_baumer_cams = baumer_system->getNumCameras();

  std::cout << "Found " << num_detected_baumer_cams << " baumer cams!\n";
  
  // these baumer cameras work only monochrome
  bool use_rgb = false;

  std::vector<baumer::BCamera*> detected_baumer_cams(num_detected_baumer_cams, nullptr);


  std::vector<std::vector<uint8_t>> baumer_image_buffers(num_detected_baumer_cams, std::vector<uint8_t>() );

  std::vector<cv::Mat> cam_images(num_detected_baumer_cams);

  for(uint32_t camera_idx = 0; camera_idx < num_detected_baumer_cams; ++camera_idx) {
    detected_baumer_cams[camera_idx] = baumer_system->getCamera(camera_idx, use_rgb);

    uint32_t current_image_buffer_size =   detected_baumer_cams[camera_idx]->getNumChannels() 
                                         * detected_baumer_cams[camera_idx]->getWidth()
                                         * detected_baumer_cams[camera_idx]->getHeight();
    
    baumer_image_buffers[camera_idx].resize(current_image_buffer_size);

    std::cout << "Allocated image buffer for " << detected_baumer_cams[camera_idx]->getNumChannels() << " channel image of size"
                                               << detected_baumer_cams[camera_idx]->getWidth() << " x " << detected_baumer_cams[camera_idx]->getHeight() 
                                               << " for camera " << camera_idx << "\n";
  

    detected_baumer_cams[camera_idx]->setGain(3.0f);
    detected_baumer_cams[camera_idx]->setExposureTime(30000.0f);


    cam_images[camera_idx] = cv::Mat( detected_baumer_cams[camera_idx]->getHeight(), detected_baumer_cams[camera_idx]->getWidth(), CV_8UC1);
  }

  //cv::Mat cam_image_1 = cv::Mat( detected_baumer_cams[0]->getHeight(), detected_baumer_cams[0]->getWidth(), CV_8UC1);
  //cv::Mat cam_image_2 = cv::Mat( detected_baumer_cams[1]->getHeight(), detected_baumer_cams[1]->getWidth(), CV_8UC1);



  uint image_counter = 1;
  while(true) {

    // capture all images 
    for(uint32_t camera_idx = 0; camera_idx < num_detected_baumer_cams; ++camera_idx) {
      
          //std::cout << " Trying to memcpy " << baumer_image_buffers[camera_idx].size() << "\n";
          boost::mutex::scoped_lock l(detected_baumer_cams[camera_idx]->getMutexLock());
          if(0 != detected_baumer_cams[camera_idx]->capture()){
              memcpy(cam_images[camera_idx].data, detected_baumer_cams[camera_idx]->capture(), baumer_image_buffers[camera_idx].size());
          }

      } 
  
      // visualize all images 
      for(uint32_t camera_idx = 0; camera_idx < num_detected_baumer_cams; ++camera_idx) {
        std::string const image_label = "image " + std::to_string(camera_idx);
        cv::imshow(image_label, cam_images[camera_idx]);
      }

      int keycode = cv::waitKey(1) & 0xFF;

      std::string image_counter_post_fix = (image_counter < 10) ?  "0" : "";

      image_counter_post_fix += std::to_string(image_counter) + ".png";

      // space
      if(32 == keycode) {
        for(uint32_t camera_idx = 0; camera_idx < num_detected_baumer_cams; ++camera_idx) {
          std::string const image_name = "cam_" + std::to_string(camera_idx) + "__im_" + image_counter_post_fix ;
            std::cout << "Wrote: " << image_name << std::endl;
            cv::imwrite("./" + image_name, cam_images[camera_idx]);

        }
        ++image_counter;
      }
  }


  return 0;
}

/*
int main(int argc, char** argv)
{
    namespace po = boost::program_options;
    po::options_description desc("Options");

    desc.add_options()
      ("help,h", "Print help messages") //bool
      ("image_0", po::value<std::string>(), "image path 0") //std::string
      ("image_1", po::value<std::string>(), "image path 1")
      ("cam_0", po::value<std::string>(), "cam 0")
      ("cam_1", po::value<std::string>(), "cam 1")
    ;

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc),  vm); // can throw
    auto img0_path = vm["image_0"].as<std::string>();
    auto img1_path = vm["image_1"].as<std::string>();
    auto cam0_path = vm["cam_0"].as<std::string>();
    auto cam1_path = vm["cam_1"].as<std::string>();

    cv::Mat cam_mat[2], R[2], T[2];
    parse_cam_info(cam0_path, cam_mat[0], R[0], T[0]);
    parse_cam_info(cam1_path, cam_mat[1], R[1], T[1]);

//    cout << "T_0: ";
//    cout << T[0] << endl;
//    cout << T[1] << endl;

    rectify_images(img0_path, img1_path, cam_mat[0], R[0], T[0], cam_mat[1], R[1], T[1]);
}
*/