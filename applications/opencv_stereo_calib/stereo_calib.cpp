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
#include <iterator>
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <boost/algorithm/string.hpp>

using namespace cv;
using namespace std;

std::shared_ptr<baumer::BSystem> baumer_system = nullptr;

void calibrate(std::vector<std::vector<cv::Point3f>> const& objectPoints, std::vector<std::vector<cv::Point2f>> const& imagePoints_l, 
               std::vector<std::vector<cv::Point2f>> const& imagePoints_r, cv::Size const& imageSize, std::vector<cv::Mat> const& images,
               bool showRectified = true, bool useCalibrated = true, const string& out_dir = "")
{
    cout << "Running stereo calibration ...\n";

    Mat cameraMatrix[2], distCoeffs[2];
    cameraMatrix[0] = initCameraMatrix2D(objectPoints,imagePoints_l,imageSize,0);
    cameraMatrix[1] = initCameraMatrix2D(objectPoints,imagePoints_r,imageSize,0);
    Mat R, T, E, F;

    double rms = stereoCalibrate(objectPoints, imagePoints_l, imagePoints_r,
                                 cameraMatrix[0], distCoeffs[0],
                                 cameraMatrix[1], distCoeffs[1],
                                 imageSize, R, T, E, F,
                                 CALIB_FIX_ASPECT_RATIO +
                                 CALIB_ZERO_TANGENT_DIST +
                                 CALIB_USE_INTRINSIC_GUESS +
                                 CALIB_SAME_FOCAL_LENGTH +
                                 CALIB_RATIONAL_MODEL +
                                 CALIB_FIX_K3 + CALIB_FIX_K4 + CALIB_FIX_K5,
                                 TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 100, 1e-5) );
    cout << "done with RMS error=" << rms << endl;

// CALIBRATION QUALITY CHECK
// because the output fundamental matrix implicitly
// includes all the output information,
// we can check the quality of calibration using the
// epipolar geometry constraint: m2^t*F*m1=0
    double err = 0;
    int npoints = 0;
    vector<Vec3f> lines[2];

    int nimages = images.size() / 2;

    for( int i = 0; i < nimages; i++ )
    {
        int npt = (int)imagePoints_l.size();
        Mat imgpt[2];

        imgpt[0] = Mat(imagePoints_l[i]);
        undistortPoints(imgpt[0], imgpt[0], cameraMatrix[0], distCoeffs[0], Mat(), cameraMatrix[0]);
        computeCorrespondEpilines(imgpt[0], 0+1, F, lines[0]);
        imgpt[1] = Mat(imagePoints_r[i]);
        undistortPoints(imgpt[1], imgpt[1], cameraMatrix[1], distCoeffs[1], Mat(), cameraMatrix[1]);
        computeCorrespondEpilines(imgpt[1], 1+1, F, lines[1]);

        for( int j = 0; j < npt; j++ )
        {
            double errij = fabs(imagePoints_l[i][j].x*lines[1][j][0] +
                                imagePoints_l[i][j].y*lines[1][j][1] + lines[1][j][2]) +
                           fabs(imagePoints_r[i][j].x*lines[0][j][0] +
                                imagePoints_r[i][j].y*lines[0][j][1] + lines[0][j][2]);
            err += errij;
        }
        npoints += npt;
    }
    cout << "average epipolar err = " <<  err/npoints << endl;

    // save intrinsic parameters
    FileStorage fs(out_dir + "/intrinsics.yml", FileStorage::WRITE);
    if( fs.isOpened() )
    {
        fs << "M1" << cameraMatrix[0] << "D1" << distCoeffs[0] <<
           "M2" << cameraMatrix[1] << "D2" << distCoeffs[1];
        fs.release();
    }
    else
        cout << "Error: can not save the intrinsic parameters\n";

    Mat R1, R2, P1, P2, Q;
    Rect validRoi[2];

    cout << "rotation matrix: " << endl;
    cout << R << endl;
    cout << "tranlsate matrix: " << endl;
    cout << T << endl;
    stereoRectify(cameraMatrix[0], distCoeffs[0],
                  cameraMatrix[1], distCoeffs[1],
                  imageSize, R, T, R1, R2, P1, P2, Q,
                  CALIB_ZERO_DISPARITY, 1, imageSize, &validRoi[0], &validRoi[1]);

    fs.open(out_dir + "/extrinsics.yml", FileStorage::WRITE);
    if( fs.isOpened() )
    {
        fs << "R" << R << "T" << T << "R1" << R1 << "R2" << R2 << "P1" << P1 << "P2" << P2 << "Q" << Q;
        fs.release();
    }
    else
        cout << "Error: can not save the extrinsic parameters\n";


    fs.open(out_dir + "/camera_params.yml", FileStorage::WRITE);
    if( fs.isOpened() )
    {
        fs << "M1" << cameraMatrix[0] << "D1" << distCoeffs[0] <<
           "M2" << cameraMatrix[1] << "D2" << distCoeffs[1] << "R" << R << "T" << T << "R1" << R1 << "R2" << R2 << "P1" << P1 << "P2" << P2 << "Q" << Q <<
           "ValidRoi1" << validRoi[0] << "ValidRoi2" << validRoi[1];
        fs.release();
    }
    else
        cout << "Error: can not save the extrinsic parameters\n";

    // OpenCV can handle left-right
    // or up-down camera arrangements
    bool isVerticalStereo = fabs(P2.at<double>(1, 3)) > fabs(P2.at<double>(0, 3));

// COMPUTE AND DISPLAY RECTIFICATION
    if( !showRectified )
        return;

    Mat rmap[2][2];
// IF BY CALIBRATED (BOUGUET'S METHOD)
    if( useCalibrated )
    {
        // we already computed everything
    }
// OR ELSE HARTLEY'S METHOD
    else
        // use intrinsic parameters of each camera, but
        // compute the rectification transformation directly
        // from the fundamental matrix
    {
        vector<Point2f> allimgpt[2];
        //for(int  k = 0; k < 2; k++ )
        //{
            for(int i = 0; i < nimages; i++ )
                std::copy(imagePoints_l[i].begin(), imagePoints_l[i].end(), back_inserter(allimgpt[0]));
            for(int i = 0; i < nimages; i++ )
                std::copy(imagePoints_r[i].begin(), imagePoints_r[i].end(), back_inserter(allimgpt[1]));
        //}
        F = findFundamentalMat(Mat(allimgpt[0]), Mat(allimgpt[1]), FM_8POINT, 0, 0);
        Mat H1, H2;
        stereoRectifyUncalibrated(Mat(allimgpt[0]), Mat(allimgpt[1]), F, imageSize, H1, H2, 3);

        R1 = cameraMatrix[0].inv()*H1*cameraMatrix[0];
        R2 = cameraMatrix[1].inv()*H2*cameraMatrix[1];
        P1 = cameraMatrix[0];
        P2 = cameraMatrix[1];
    }

    //Precompute maps for cv::remap()
    initUndistortRectifyMap(cameraMatrix[0], distCoeffs[0], R1, P1, imageSize, CV_16SC2, rmap[0][0], rmap[0][1]);
    initUndistortRectifyMap(cameraMatrix[1], distCoeffs[1], R2, P2, imageSize, CV_16SC2, rmap[1][0], rmap[1][1]);

//    cout << "rmap00: " << endl;
//    cout << rmap[0][0] << endl;

    Mat canvas;
    double sf;
    int w, h;
    if( !isVerticalStereo )
    {
        sf = 600./MAX(imageSize.width, imageSize.height);
        w = cvRound(imageSize.width*sf);
        h = cvRound(imageSize.height*sf);
        canvas.create(h, w*2, CV_8UC3);
    }
    else
    {
        sf = 300./MAX(imageSize.width, imageSize.height);
        w = cvRound(imageSize.width*sf);
        h = cvRound(imageSize.height*sf);
        canvas.create(h*2, w, CV_8UC3);
    }

    for(int i = 0; i < nimages; i++ )
    {
        for(int k = 0; k < 2; k++ )
        {
            //Mat img = imread(goodImageList[i*2+k], 0), rimg, cimg;
            Mat img, rimg, cimg;
            img = images[i*2 + k];
            remap(img, rimg, rmap[k][0], rmap[k][1], INTER_LINEAR);
            cvtColor(rimg, cimg, COLOR_GRAY2BGR);
            Mat canvasPart = !isVerticalStereo ? canvas(Rect(w*k, 0, w, h)) : canvas(Rect(0, h*k, w, h));
            resize(cimg, canvasPart, canvasPart.size(), 0, 0, INTER_AREA);
            //if( useCalibrated )
            //{
                Rect vroi(cvRound(validRoi[k].x*sf), cvRound(validRoi[k].y*sf),
                          cvRound(validRoi[k].width*sf), cvRound(validRoi[k].height*sf));
                rectangle(canvasPart, vroi, Scalar(0,0,255), 3, 8);
            //}
        }

        if( !isVerticalStereo )
            for(int j = 0; j < canvas.rows; j += 16 )
                line(canvas, Point(0, j), Point(canvas.cols, j), Scalar(0, 255, 0), 1, 8);
        else
            for(int j = 0; j < canvas.cols; j += 16 )
                line(canvas, Point(j, 0), Point(j, canvas.rows), Scalar(0, 255, 0), 1, 8);
        imshow("rectified", canvas);
        char c = (char)waitKey();
        if( c == 27 || c == 'q' || c == 'Q' )
            break;
    }
}

void initBaumerSystem(bool useRGB, baumer::BCamera* &leftCamera, baumer::BCamera* &rightCamera, float gain)
{
    baumer_system = std::make_shared<baumer::BSystem>();
    baumer_system->init();
    uint32_t num_baumer_cams = baumer_system->getNumCameras();
    std::cout << num_baumer_cams << " cameras detected." << std::endl;
    if(num_baumer_cams != 2)  {
        std::cerr << "Error. Please connect exactly 2 cameras to the system!" << std::endl;
        std::exit(1);
    }
    leftCamera = baumer_system->getCamera(0, useRGB);
    leftCamera->setGain(gain);
    rightCamera = baumer_system->getCamera(1, useRGB);
    rightCamera->setGain(gain);
}

void cleanupBaumerSystem(baumer::BCamera* &leftCamera, baumer::BCamera* &rightCamera) 
{
}

void 
stream(bool useRGB = false) 
{
    baumer::BCamera* leftCamera;
    baumer::BCamera* rightCamera;
    initBaumerSystem(useRGB, leftCamera, rightCamera, 3.0f);

    std::vector<std::vector<uint8_t>> image_buffers(2, std::vector<uint8_t>());
    uint32_t image_buffer_size = leftCamera->getNumChannels()
                                 * leftCamera->getWidth()
                                 * leftCamera->getHeight();
    image_buffers[0].resize(image_buffer_size);
    image_buffers[1].resize(image_buffer_size);
    
    cv::Mat cam_image_1, cam_image_2;
    if(useRGB) {
        cam_image_1 = cv::Mat( leftCamera->getHeight(), leftCamera->getWidth(), CV_8UC3 );
        cam_image_2 = cv::Mat( rightCamera->getHeight(), rightCamera->getWidth(), CV_8UC3 );
    } else {
        cam_image_1 = cv::Mat( leftCamera->getHeight(), leftCamera->getWidth(), CV_8UC1 );
        cam_image_2 = cv::Mat( rightCamera->getHeight(), rightCamera->getWidth(), CV_8UC1 );
    }

    while(true) {
        {
            boost::mutex::scoped_lock l(leftCamera->getMutexLock());
            if(0 != leftCamera->capture()) 
                memcpy(cam_image_1.data, leftCamera->capture(), image_buffers[0].size());
        }
        {
            boost::mutex::scoped_lock r(rightCamera->getMutexLock());
            if(0 != rightCamera->capture()) 
                memcpy(cam_image_2.data, rightCamera->capture(), image_buffers[0].size());

        }

        if(useRGB) {
            cv::cvtColor(cam_image_1, cam_image_1, cv::COLOR_RGB2BGR);
            cv::cvtColor(cam_image_2, cam_image_2, cv::COLOR_RGB2BGR);
        }

        cv::imshow("Left", cam_image_1);
        cv::imshow("Right", cam_image_2);
        int keycode = cv::waitKey(1) & 0xFF;
        if(27 == keycode || 'q' == keycode || 'Q' == keycode) {
            break;
        }
    }

}

void
calibrateStream(Size boardSize, float squareSize, bool displayCorners = false, bool useRGB = false, float gain = 3.0f) 
{
    baumer::BCamera* leftCamera;
    baumer::BCamera* rightCamera;
    initBaumerSystem(useRGB, leftCamera, rightCamera, gain);

    std::vector<std::vector<uint8_t>> image_buffers(2, std::vector<uint8_t>());
    uint32_t image_buffer_size = leftCamera->getNumChannels()
                                 * leftCamera->getWidth()
                                 * leftCamera->getHeight();
    image_buffers[0].resize(image_buffer_size);
    image_buffers[1].resize(image_buffer_size);
    
    cv::Mat cam_image_1, cam_image_2, cam_image_1_color, cam_image_2_color;
    if(useRGB) {
        cam_image_1_color = cv::Mat( leftCamera->getHeight(), leftCamera->getWidth(), CV_8UC3 );
        cam_image_2_color = cv::Mat( rightCamera->getHeight(), rightCamera->getWidth(), CV_8UC3 );
        cam_image_1 = cv::Mat( leftCamera->getHeight(), leftCamera->getWidth(), CV_8UC1 );
        cam_image_2 = cv::Mat( rightCamera->getHeight(), rightCamera->getWidth(), CV_8UC1 );
    } else {
        cam_image_1 = cv::Mat( leftCamera->getHeight(), leftCamera->getWidth(), CV_8UC1 );
        cam_image_2 = cv::Mat( rightCamera->getHeight(), rightCamera->getWidth(), CV_8UC1 );
    }

    std::vector<std::vector<cv::Point2f> > imagePoints_l;
    std::vector<std::vector<cv::Point2f> > imagePoints_r;
    std::vector<std::vector<cv::Point3f> > objectPoints;
    std::vector<cv::Mat> images; 

    Size imageSize;
    imageSize.width = leftCamera->getWidth();
    imageSize.height = leftCamera->getHeight();

    std::cout << "Press SPACE to register calibration image." << std::endl;
    std::cout << "Press Q to stop calibration registration." << std::endl;

    std::size_t count = 0;
    
    while(true) {
        {
            boost::mutex::scoped_lock l(leftCamera->getMutexLock());
            if(0 != leftCamera->capture()) 
                if(useRGB)
                    memcpy(cam_image_1_color.data, leftCamera->capture(), image_buffers[0].size());
                else
                    memcpy(cam_image_1.data, leftCamera->capture(), image_buffers[0].size());
        }
        {
            boost::mutex::scoped_lock r(rightCamera->getMutexLock());
            if(0 != rightCamera->capture()) 
                if(useRGB)
                    memcpy(cam_image_2_color.data, rightCamera->capture(), image_buffers[0].size());
                else
                    memcpy(cam_image_2.data, rightCamera->capture(), image_buffers[0].size());

        }

        if(useRGB) {
            cv::cvtColor(cam_image_1_color, cam_image_1, cv::COLOR_BGR2GRAY);
            cv::cvtColor(cam_image_2_color, cam_image_2, cv::COLOR_BGR2GRAY);
        }

        std::vector<cv::Point2f> corners_l;
        std::vector<cv::Point2f> corners_r;
        bool found_l = findChessboardCorners(cam_image_1, boardSize, corners_l,
                                             CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE);
        bool found_r = findChessboardCorners(cam_image_2, boardSize, corners_r,
                                             CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE);


        if(found_l && found_r) {
            // perform corner enhancement
            cornerSubPix(cam_image_1, corners_l, cv::Size(11, 11), cv::Size(-1, -1),
                         cv::TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 30, 0.01));
            cornerSubPix(cam_image_2, corners_r, cv::Size(11, 11), cv::Size(-1, -1),
                         cv::TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 30, 0.01));

            cv::Mat cimg_l, cimg_r;
            if(!useRGB) {
                cv::cvtColor(cam_image_1, cimg_l, COLOR_GRAY2BGR);
                cv::cvtColor(cam_image_2, cimg_r, COLOR_GRAY2BGR);
            } else {
                cimg_l = cam_image_1;
                cimg_r = cam_image_2;
            }
            drawChessboardCorners(cimg_l, boardSize, corners_l, found_l);
            drawChessboardCorners(cimg_r, boardSize, corners_r, found_r);
            cv::imshow("Left", cimg_l);
            cv::imshow("Right", cimg_r);
            int keycode = cv::waitKey(1) & 0xFF;
            if(32 == keycode) {
                std::cout << "Registering image pair" << std::endl;    
                imagePoints_l.push_back(corners_l);
                imagePoints_r.push_back(corners_r);
                images.push_back(cam_image_1.clone());
                images.push_back(cam_image_2.clone());
                count++;
            }
            if(27 == keycode || 'q' == keycode || 'Q' == keycode) {
                break;
            }
        } else {
            cv::imshow("Left", cam_image_1);
            cv::imshow("Right", cam_image_2);
            cv::waitKey(1);
        }
        
        
    }

    objectPoints.resize(images.size() / 2);
    for(int i = 0; i < count; i++ )
    {
        for(int j = 0; j < boardSize.height; j++ )
            for(int k = 0; k < boardSize.width; k++ )
                objectPoints[i].push_back(Point3f(k*squareSize, j*squareSize, 0));
    }

    calibrate(objectPoints, imagePoints_l, imagePoints_r, imageSize, images, true, true, "./");


    cleanupBaumerSystem(leftCamera, rightCamera); 

    return;
}

void
calibrateImages(const vector<string>& imagelist, Size boardSize, float squareSize, bool displayCorners = false, bool useCalibrated=true, bool showRectified=true, const string& out_dir = "")
{
    if( imagelist.size() % 2 != 0 )
    {
        cout << "Error: the image list contains odd (non-even) number of elements\n";
        return;
    }

    const int maxScale = 2;
    // ARRAY AND VECTOR STORAGE:

    vector<vector<Point2f> > imagePoints[2];
    vector<vector<Point3f> > objectPoints;
    Size imageSize;

    int i, j, k, nimages = (int)imagelist.size()/2;

    imagePoints[0].resize(nimages);
    imagePoints[1].resize(nimages);
    vector<Mat> images;
    //vector<string> goodImageList;

    for( i = j = 0; i < nimages; i++ )
    {
        cv::Mat left, right;
        for( k = 0; k < 2; k++ )
        {
            const string& filename = imagelist[i*2+k];
            Mat img = imread(filename, 0);
            if(img.empty())
                break;
            if( imageSize == Size() )
                imageSize = img.size();
            else if( img.size() != imageSize )
            {
                cout << "The image " << filename << " has the size different from the first image size. Skipping the pair\n";
                break;
            }
            bool found = false;
            vector<Point2f>& corners = imagePoints[k][j];
            for( int scale = 1; scale <= maxScale; scale++ )
            {
                Mat timg;
                if( scale == 1 )
                    timg = img;
                else
                    resize(img, timg, Size(), scale, scale, INTER_LINEAR_EXACT);
                found = findChessboardCorners(timg, boardSize, corners,
                                              CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE);
                if( found )
                {
                    if( scale > 1 )
                    {
                        Mat cornersMat(corners);
                        cornersMat *= 1./scale;
                    }
                    break;
                }
            }
            if( displayCorners )
            {
                //cout << filename << endl;
                Mat cimg, cimg1;
                cvtColor(img, cimg, COLOR_GRAY2BGR);
                drawChessboardCorners(cimg, boardSize, corners, found);
                double sf = 640./MAX(img.rows, img.cols);
                resize(cimg, cimg1, Size(), sf, sf, INTER_LINEAR_EXACT);
                imshow("corners", cimg1);
                char c = (char)waitKey(500);
                if( c == 27 || c == 'q' || c == 'Q' ) //Allow ESC to quit
                    exit(-1);
            }
            else
                putchar('.');
            if( !found )
                break;
            cornerSubPix(img, corners, Size(11,11), Size(-1,-1),
                         TermCriteria(TermCriteria::COUNT+TermCriteria::EPS,
                                      30, 0.01));
            if(k == 0) 
                left = img.clone();
            if(k == 1)
                right = img.clone();
        }
        if( k == 2 )
        {
            images.push_back(left);
            images.push_back(right);
            //goodImageList.push_back(imagelist[i*2]);
            //goodImageList.push_back(imagelist[i*2+1]);
            j++;
        }
    }
    cout << j << " pairs have been successfully detected.\n";
    nimages = j;
    if( nimages < 2 )
    {
        cout << "Error: too little pairs to run the calibration\n";
        return;
    }

    imagePoints[0].resize(nimages);
    imagePoints[1].resize(nimages);
    objectPoints.resize(nimages);

    for( i = 0; i < nimages; i++ )
    {
        for( j = 0; j < boardSize.height; j++ )
            for( k = 0; k < boardSize.width; k++ )
                objectPoints[i].push_back(Point3f(k*squareSize, j*squareSize, 0));
    }

    calibrate(objectPoints, imagePoints[0], imagePoints[1], imageSize, images, showRectified, useCalibrated, out_dir);
//    {
//        Mat camMat_est;
//        Mat distCoeffs_est;
//        vector<Mat> rvecs, tvecs;
//        cout << "Calibrating..." << endl;
//        double rep_err = calibrateCamera(objectPoints, imagePoints[0], imageSize, camMat_est, distCoeffs_est, rvecs, tvecs);
//        cout << camMat_est << endl;
//        for (auto &t : tvecs)
//        {
//            cout << t << endl;
//        }
//    }
}


bool readStringList( const string& filename, vector<string>& l )
{
    l.resize(0);
    FileStorage fs(filename, FileStorage::READ);
    if( !fs.isOpened() )
        return false;
    FileNode n = fs.getFirstTopLevelNode();
    if( n.type() != FileNode::SEQ )
        return false;
    FileNodeIterator it = n.begin(), it_end = n.end();
    for( ; it != it_end; ++it )
        l.push_back((string)*it);
    return true;
}

int main(int argc, char** argv)
{
    Size boardSize;
    string imagelistfn;
    bool showRectified;
    cv::CommandLineParser parser(argc, argv, "{columns c|7|Number of inner horizontal corners}"
                                             "{rows r|5|Number of inner vertical corners}"
                                             "{scale s|1.0|Scale}"
                                             "{show nr||}"
                                             "{live l||We do it live! (with a Baumer camera setup)}"
                                             "{test t||Testing mode of camera streaming}"
                                             "{usergb u||Use RGB (Baumer camera setup)}"
                                             "{help h usage ?||print this message}"
                                             "{@input|stereo_calib.xml|Input XML file list}");
    if (parser.has("help")) {
        parser.printMessage();
        return 0;
    }

    if (!parser.check())
    {
        parser.printErrors();
        return 1;
    }

    showRectified = parser.has("show");
    boardSize.width = parser.get<int>("c");
    boardSize.height = parser.get<int>("r");
    float squareSize = parser.get<float>("s");

    if(parser.has("test")) {
        std::cout << "Test mode" << std::endl;
        bool useRGB = parser.has("usergb");
        stream(useRGB);
        return 0;
    }

    if (parser.has("live")) {
        std::cout << "Live calibration" << std::endl;
        bool useRGB = parser.has("usergb");
        calibrateStream(boardSize, squareSize, true, useRGB); 
        return 0;
    }

    auto xml_path = parser.get<string>("@input");
    imagelistfn = samples::findFile(xml_path);

    vector<string> imagelist;
    bool ok = readStringList(imagelistfn, imagelist);
    if(!ok || imagelist.empty())
    {
        cout << "can not open " << imagelistfn << " or the string list is empty" << endl;
        parser.printMessage();
        return 1;
    }


    string dir = xml_path.substr(0, xml_path.find_last_of("/\\"));
    for (string &path : imagelist)
        path = string(dir + "/" + path);

    calibrateImages(imagelist, boardSize, squareSize, true, true, showRectified, dir);
    return 0;
    }
