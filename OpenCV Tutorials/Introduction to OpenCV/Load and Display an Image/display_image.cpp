
#include <opencv4/opencv2/core.hpp>
// core section, as here are defined the basic building blocks of the library
#include <opencv4/opencv2/imgcodecs.hpp>
#include <opencv4/opencv2/highgui.hpp>
// highgui module, as this contains the functions for input and output operations
#include <iostream>
using namespace cv;
// To avoid data structure and function name conflicts with other libraries, OpenCV has its own namespace: cv.
using namespace std;

int main( int argc, char** argv )
{
    String imageName( "HappyFish.jpg" ); // by default
    if( argc > 1)
    {
        imageName = argv[1];
    }
    Mat image;
    image = imread( samples::findFile( imageName ), IMREAD_UNCHANGED ); // Read the file

    // The second argument specifies the format in what we want the image
    // 可取值为
    // IMREAD_UNCHANGED (<0) loads the image as is (including the alpha channel if present)
    // IMREAD_GRAYSCALE ( 0) loads the image as an intensity one
    // IMREAD_COLOR (>0) loads the image in the RGB format

    // OpenCV 自带的支持的图像格式：
    // Windows bitmap (bmp)
    // portable image formats (pbm, pgm, ppm)
    // Sun raster (sr, ras)
    // 在插件的帮助下 (you need to specify to use them if you build yourself the library,
    // nevertheless in the packages we ship present by default) 也可以加载图像格式，如:
    // JPEG (jpeg, jpg, jpe), JPEG 2000 (jp2 - codenamed in the CMake as Jasper), TIFF files (tiff, tif) and portable network graphics (png).
    // Furthermore, OpenEXR is also a possibility.

    if( image.empty() )                      // Check for invalid input
    {
        cout <<  "Could not open or find the image" << std::endl ;
        return -1;
    }
    namedWindow( "Display window", WINDOW_AUTOSIZE ); // Create a window for display.
    // These are automatically managed by OpenCV once you create them.
    // For this you need to specify its name and how it should handle the change of
    // the image it contains from a size point of view. It may be:
    // WINDOW_AUTOSIZE  is the only supported one if you do not use the Qt backend.
    //                  In this case the window size will take up the size of the image it shows.
    //                  No resize permitted!
    // WINDOW_NORMAL    on Qt you may use this to allow window resize.
    //                  The image will resize itself according to the current window size.
    //                  By using the | operator you also need to specify if you would like the image
    //                  to keep its aspect ratio (WINDOW_KEEPRATIO) or not (WINDOW_FREERATIO).
    imshow( "Display window", image );                // Show our image inside it.
    // Update the content of the OpenCV window
    waitKey(0); // Wait for a keystroke in the window
    return 0;
}
