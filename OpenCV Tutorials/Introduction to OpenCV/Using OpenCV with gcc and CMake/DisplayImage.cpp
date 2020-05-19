#include <stdio.h>
#include <opencv4/opencv2/opencv.hpp>

using namespace cv;

int main(int argc, char** argv)
{
    String imageName("foo.jpg"); // by default
    if (argc != 2)
    {
        printf("usage: DisplayImage <Image_Path>\n");
        return -1;
    }

    Mat image;
    image = imread(argv[1], 1);
    // imageName = argv[1] image = imread(samples::findFile(imageName), IMREAD_COLOR); // Read the file

    if ( !image.data )
    {
        printf("No image data \n");
        return -1;
    }
    namedWindow("Display Image", WINDOW_AUTOSIZE); // Create a window for display
    imshow("Display Image", image); // Show our image inside it.

    waitKey(0); // Wait for a keystroke in the window
    return 0;
}
