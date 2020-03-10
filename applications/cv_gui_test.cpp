#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <string>
using namespace cv;
using namespace std;
int main( int argc, char** argv )
{
    string imageName("./images/Baby3/view1.png"); // by default
    if( argc > 1)
    {
        imageName = argv[1];
    }
    Mat image;
    image = imread(imageName.c_str(), IMREAD_COLOR); // Read the file
    if( image.empty() )                      // Check for invalid input
    {
        cout <<  "Could not open or find the image" << std::endl ;
        return -1;
    }
    
    namedWindow( "Display window", WINDOW_AUTOSIZE ); // Create a window for display.
    
    while(true) {
        imshow( "Display window", image );                // Show our image inside it.
        unsigned char k = waitKey(10) & 0xFF;
        std::cout << "Refresh\n";
    }
    //waitKey(0); // Wait for a keystroke in the window
    return 0;
}
