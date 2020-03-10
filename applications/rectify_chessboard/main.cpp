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

void rectify_images(const string &img0_path, const string &img1_path,
                    cv::Mat &cam_0, cv::Mat &Rot_0, cv::Mat &T_0,
                    cv::Mat &cam_1, cv::Mat &Rot_1, cv::Mat &T_1)
{
    cv::Mat img0 = cv::imread(img0_path);
    cv::Mat img1 = cv::imread(img1_path);
    assert (img0.size() == img1.size());

    //https://dsp.stackexchange.com/questions/19330/how-to-get-relative-rotation-matrix-from-two-orientation-values-in-android
    cv::Mat Rot = Rot_1 * Rot_0.inv();
    cv::Mat T =  -Rot * T_0 + T_1;

    cv::Mat dist_coeffs[2]; //dummy distor coefficients

    assert(T.type() == T_0.type());
    assert(T.type() == T_1.type());

    Mat R1, R2, P1, P2, Q;
    Rect validRoi[2];
    cv::Size img_size = img0.size();
    cv::stereoRectify(cam_0, dist_coeffs[0], cam_1, dist_coeffs[0],
                  img_size, Rot, T, R1, R2, P1, P2, Q,
                  CALIB_ZERO_DISPARITY, 1, img_size, &validRoi[0], &validRoi[1]);

    cout << "rectification R1: " << endl << R1 << endl;
    cout << "rectification R2: " << endl << R2 << endl;

    Mat rmap[2][2];
    initUndistortRectifyMap(cam_0, dist_coeffs[0], R1, P1, img_size, CV_16SC2, rmap[0][0], rmap[0][1]);
    initUndistortRectifyMap(cam_1, dist_coeffs[1], R2, P2, img_size, CV_16SC2, rmap[1][0], rmap[1][1]);

    double sf;
    int w, h;
    Mat canvas;
    sf = 600./MAX(img_size.width, img_size.height);
    w = cvRound(img_size.width*sf);
    h = cvRound(img_size.height*sf);
    canvas.create(h, w*2, CV_8UC3);

    for (int k = 0; k < 2; ++k)
    {
        Mat rect_img;
        if (k == 0)
            remap(img0, rect_img, rmap[k][0], rmap[k][1], INTER_LINEAR);
        else
            remap(img1, rect_img, rmap[k][0], rmap[k][1], INTER_LINEAR);

        Mat canvasPart = canvas(Rect(w*k, 0, w, h));
        resize(rect_img, canvasPart, canvasPart.size(), 0, 0, INTER_AREA);
        Rect vroi(cvRound(validRoi[k].x*sf), cvRound(validRoi[k].y*sf),
                  cvRound(validRoi[k].width*sf), cvRound(validRoi[k].height*sf));
        rectangle(canvasPart, vroi, Scalar(0,0,255), 3, 8);
    }

    for(int j = 0; j < canvas.rows; j += 16 )
        line(canvas, Point(0, j), Point(canvas.cols, j), Scalar(0, 255, 0), 1, 8);
    imshow("rectified", canvas);
    char c = (char)waitKey();
    if( c == 27 || c == 'q' || c == 'Q' )
        return;
}


vector<float> parse_number(vector<string> &candidates)
{
    vector<float> numbers;
    for (auto &str : candidates)
    {
        float n;
        if (bool(stringstream(str) >> n))
        {
            numbers.push_back(n);
        }
    }
    return numbers;
}

//http://perso.lcpc.fr/tarel.jean-philippe/syntim/paires.html
void parse_cam_info(const std::string &path, cv::Mat &out_intrinsic_mat, cv::Mat &out_R, cv::Mat &out_T)
{
//    Intrinsic parameters:
//    Image center: u0 = 370.852356, v0 = 296.351501
//    Scale factor:
//    au = 1707.133789, av = 1656.258789
//    Image size: dimx = 768, dimy = 576
//    Focale: 1.0
//    Extrinsic parameters:
//    Rotation:
//    {{0.609094, -0.198737, 0.767794},
//     {-0.004385, 0.967236, 0.253839},
//     {-0.793086, -0.157979, 0.588266}}
//    Translation:
//    {-1068.461182, -214.166428, -1126.530640}
//
    std::ifstream ifs(path);
    std::vector<std::string> lines;
    std::string line;
    while(std::getline(ifs, line))
    {
        if (line.empty())
            continue;
        lines.push_back(line);
        line.clear();
    }

    //for (std::string &x : lines) {
    //    std::cout << x << std::endl;
    //}

    //https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html?highlight=camera%20calibration
    cv::Mat intrinsic_mat = cv::Mat::eye(cv::Size(3,3), CV_64F);
    cv::Mat R = cv::Mat::zeros(cv::Size(3,3), CV_64F);
    cv::Mat T = cv::Mat::zeros(cv::Size(1,3), CV_64F);
    for (size_t i = 0; i < lines.size(); ++i){
        string &x = lines[i];
        //std::cout << x << std::endl;
        vector<string> components;
        boost::replace_all(x, ",", "");
        boost::algorithm::split(components, x, boost::is_any_of(" "));

        if (x.find("Image center") != std::string::npos){
            vector<float> numbers = parse_number(components);
            assert(numbers.size() == 2);
            float u0 = numbers[0];
            float v0 = numbers[1];
            cout << "u0, v0: " << u0 << " " << v0 << endl;
            intrinsic_mat.at<double>(1,2) = v0;
            intrinsic_mat.at<double>(0,2) = u0;
        }
        else if(x[0] == 'a' && x[1] == 'u'){
            vector<float> numbers = parse_number(components);
            assert(numbers.size() == 2);
            float au = numbers[0];
            float av = numbers[1];
            cout <<"au, av: " << au  << " " << av << endl;
            intrinsic_mat.at<double>(0,0) = au;
            intrinsic_mat.at<double>(1,1) = av;
        }
        else if(x.find("Image size") != string::npos){
            vector<float> numbers = parse_number(components);
            assert(numbers.size() == 2);
            int dx = int(numbers[0]);
            int dy = int(numbers[1]);
            cout <<"dimx, dimy: " << dx << " " << dy << endl;
        }
        else if (x.find("Focale") != string::npos)
        {
            vector<float> numbers = parse_number(components);
            assert(numbers.size() == 1);
            float focale = numbers[0];
            cout <<"Focale: " << focale << endl;
        }
        else if (x.find("Rotation") != string::npos)
        {
            vector<float> rot_values;
            for (size_t k = 1; k<4; ++k)
            {
                auto &row_str = lines[i+k];
                boost::replace_all(row_str, "{", "");
                boost::replace_all(row_str, "}", "");
                boost::replace_all(row_str, ",", "");
                vector<string> row_comps;
                boost::replace_all(x, ",", "");
                boost::algorithm::split(row_comps, row_str, boost::is_any_of(" "));

                vector<float> nums = parse_number(row_comps);
                assert(nums.size() == 3);
                rot_values.insert(rot_values.end(), nums.begin(), nums.end());
            }
            cout << "rotation matrix: ";
            for (auto &v : rot_values)
                cout << v << " ";
            cout << endl;

            for(size_t r = 0; r <3; ++r)
            {
                for(size_t c = 0; c <3; ++c)
                {
                    R.at<double>(r,c) = rot_values[r*3+c];
                }
            }
        }
        else if (x.find("Translation") != string::npos)
        {
            auto &row_str = lines[i+1];
            boost::replace_all(row_str, "{", "");
            boost::replace_all(row_str, "}", "");
            boost::replace_all(row_str, ",", "");
            vector<string> row_comps;
            boost::replace_all(x, ",", "");
            boost::algorithm::split(row_comps, row_str, boost::is_any_of(" "));
            vector<float> translate = parse_number(row_comps);
            assert(translate.size() == 3);

            cout << "tranlsation: ";
            for (auto &v : translate)
                cout << v << " ";
            cout << endl;

            T.at<double>(0,0) = translate[0];
            T.at<double>(1,0) = translate[1];
            T.at<double>(2,0) = translate[2];
        }
    }

    cout << "intrinsic mat: "<< endl;
    cout << intrinsic_mat << endl;
    cout << "rotation mat: "<< endl;
    cout << R << endl;
    cout << "translation mat: "<< endl;
    cout << T << endl;

    out_intrinsic_mat = intrinsic_mat;

    //http://perso.lcpc.fr/tarel.jean-philippe/syntim/calib.html
    out_R =  R.t();
    out_T = -R.t() * T;
}


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
