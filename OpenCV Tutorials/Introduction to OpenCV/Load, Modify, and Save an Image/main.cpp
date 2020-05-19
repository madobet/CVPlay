//! 官方的例子里有 bugs
//! args 的判断和 image 的判断必须分开
#include <opencv4/opencv2/opencv.hpp>

using namespace cv;

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        printf("usage: DisplayImage <Image_Path>\n");
        return -1;
    }

    char *imageName = argv[1];
    Mat image;
    image = imread(imageName, IMREAD_COLOR);
    if (!image.data)
    {
        printf(" No image data \n ");
        // cout << "No image data" << endl;
        return -1;
    }

    Mat gray_image;
    // convert our image from BGR to Grayscale format.
    // OpenCV has a really nice function to do this kind of transformations:
    cvtColor(image, gray_image, COLOR_BGR2GRAY);
    // we use COLOR_BGR2GRAY (because of cv::imread has BGR default channel order in case of color images).
    imwrite("../../images/Gray_Image.jpg", gray_image);

    // 开两个窗口对比显示
    namedWindow(imageName, WINDOW_AUTOSIZE);
    namedWindow("Gray image", WINDOW_AUTOSIZE);

    // OpenCV 怎么知道应该显示在哪个窗口上？
    imshow(imageName, image);
    imshow("Gray image", gray_image);
    waitKey(0);
    return 0;
}
