Mat - The Basic Image Container
===============================

**Next Tutorial:** [How to scan images, lookup tables and time measurement with OpenCV](https://docs.opencv.org/4.3.0/db/da5/tutorial_how_to_scan_images.html)

Goal
----

Multiple ways to acquire digital images from the real world:
 - digital cameras
 - scanners
 - computed tomography
 - magnetic resonance imaging
 - ...

如何数字化为图像？

![](https://docs.opencv.org/4.3.0/MatBasicImageForComputer.jpg)

上面的汽车镜像就是含有像素点强度的矩阵。根据需要以不同形式存储像素值，but in the end all images inside a computer world may be reduced to numerical matrices and other information describing the matrix itself. _OpenCV_ is a computer vision library whose main focus is to process and manipulate this information. 因此我们要讲讲 OpenCV 如何储存和处理图像。

Mat
---

OpenCV has been around since 2001. In those days the library was built around a _C_ interface and to store the image in the memory they used a C structure called _IplImage_. 一些老教程老教材里可见。 The problem with this is that it brings to the table 推上台面 all the minuses of the C language. 最大的问题是手动内存管理 manual memory management. It builds on the assumption that the user is responsible for taking care of memory allocation and deallocation. While this is not a problem with smaller programs, once your code base grows it will be more of a struggle to handle all this rather than focusing on solving your development goal.

Luckily C++ came around and introduced the concept of classes making easier for the user through automatic memory management (more or less). The good news is that C++ is fully compatible with C so no compatibility issues can arise from making the change. Therefore, OpenCV 2.0 introduced a new C++ interface which offered a new way of doing things which means you do not need to fiddle with memory management, making your code concise (less to write, to achieve more). The main downside of the C++ interface is that many embedded development systems at the moment support only C. Therefore, unless you are targeting embedded platforms, there's no point to using the _old_ methods (unless you're a masochist programmer and you're asking for trouble).

_Mat_ 不再需要手动分配和释放。 While doing this is still a possibility, most of the OpenCV functions will allocate its output data automatically. 有一点很好的是，如果传递一个已经存在的 _Mat_ 对象给函数, 那么这个对象会 reuse——如果已分配的内存满足矩阵的需要的空间的话. In other words we use at all times only as much memory as we need to perform the task.

_Mat_ is basically a class with two data parts: the matrix header (containing information such as the size of the matrix, the method used for storing, at which address is the matrix stored, and so on) and a pointer to the matrix containing the pixel values (taking any dimensionality depending on the method chosen for storing) . The matrix header size is constant, however the size of the matrix itself may vary from image to image and usually is larger by orders of magnitude.
_Mat_ 两个部分组成的类：矩阵头部（包含矩阵大小之类的信息，用于存储的方法，矩阵存储的位置，等）

OpenCV is an image processing library. It contains a large collection of image processing functions. To solve a computational challenge, most of the time you will end up using multiple functions of the library. Because of this, passing images to functions is a common practice. We should not forget that we are talking about image processing algorithms, which tend to be quite computational heavy. The last thing we want to do is further decrease the speed of your program by making unnecessary copies of potentially _large_ images.

To tackle this issue OpenCV uses a reference counting system. The idea is that each _Mat_ object has its own header, however a matrix may be shared between two _Mat_ objects by having their matrix pointers point to the same address. Moreover, the copy operators **will only copy the headers** and the pointer to the large matrix, not the data itself.

All the above objects, in the end, point to the same single data matrix and making a modification using any of them will affect all the other ones as well. In practice the different objects just provide different access methods to the same underlying data. Nevertheless, their header parts are different. The real interesting part is that you can create headers which refer to only a subsection of the full data. For example, to create a region of interest (_ROI_) in an image you just create a new header with the new boundaries:

Mat D (A, [Rect](https://docs.opencv.org/4.3.0/dc/d84/group__core__basic.html#ga11d95de507098e90bad732b9345402e8)(10, 10, 100, 100) );

Mat E = A(Range::all(), Range(1,3));

Now you may ask – if the matrix itself may belong to multiple _Mat_ objects who takes responsibility for cleaning it up when it's no longer needed. The short answer is: the last object that used it. This is handled by using a reference counting mechanism. Whenever somebody copies a header of a _Mat_ object, a counter is increased for the matrix. Whenever a header is cleaned, this counter is decreased. When the counter reaches zero the matrix is freed. Sometimes you will want to copy the matrix itself too, so OpenCV provides [cv::Mat::clone()](https://docs.opencv.org/4.3.0/d3/d63/classcv_1_1Mat.html#adff2ea98da45eae0833e73582dd4a660) and [cv::Mat::copyTo()](https://docs.opencv.org/4.3.0/d3/d63/classcv_1_1Mat.html#a33fd5d125b4c302b0c9aa86980791a77) functions.

Mat F = A.clone();

Mat G;

A.copyTo(G);

Now modifying _F_ or _G_ will not affect the matrix pointed by the _A_'s header. What you need to remember from all this is that:

*   Output image allocation for OpenCV functions is automatic (unless specified otherwise).
*   You do not need to think about memory management with OpenCV's C++ interface.
*   The assignment operator and the copy constructor only copies the header.
*   The underlying matrix of an image may be copied using the [cv::Mat::clone()](https://docs.opencv.org/4.3.0/d3/d63/classcv_1_1Mat.html#adff2ea98da45eae0833e73582dd4a660) and [cv::Mat::copyTo()](https://docs.opencv.org/4.3.0/d3/d63/classcv_1_1Mat.html#a33fd5d125b4c302b0c9aa86980791a77) functions.

Storing methods
---------------

This is about how you store the pixel values. You can select the color space and the data type used. The color space refers to how we combine color components in order to code a given color. The simplest one is the grayscale where the colors at our disposal are black and white. The combination of these allows us to create many shades of gray.

For _colorful_ ways we have a lot more methods to choose from. Each of them breaks it down to three or four basic components and we can use the combination of these to create the others. The most popular one is RGB, mainly because this is also how our eye builds up colors. Its base colors are red, green and blue. To code the transparency of a color sometimes a fourth element: alpha (A) is added.

There are, however, many other color systems each with their own advantages:

*   RGB is the most common as our eyes use something similar, however keep in mind that OpenCV standard display system composes colors using the BGR color space (red and blue channels are swapped places).
*   The HSV and HLS decompose colors into their hue, saturation and value/luminance components, which is a more natural way for us to describe colors. You might, for example, dismiss the last component, making your algorithm less sensible to the light conditions of the input image.
*   YCrCb is used by the popular JPEG image format.
*   CIE L\*a\*b* is a perceptually uniform color space, which comes in handy if you need to measure the _distance_ of a given color to another color.

Each of the building components has its own valid domains. This leads to the data type used. How we store a component defines the control we have over its domain. The smallest data type possible is _char_, which means one byte or 8 bits. This may be unsigned (so can store values from 0 to 255) or signed (values from -127 to +127). Although in case of three components this already gives 16 million possible colors to represent (like in case of RGB) we may acquire an even finer control by using the float (4 byte = 32 bit) or double (8 byte = 64 bit) data types for each component. Nevertheless, remember that increasing the size of a component also increases the size of the whole picture in the memory.

Creating a Mat object explicitly
--------------------------------

In the [Load, Modify, and Save an Image](https://docs.opencv.org/4.3.0/db/d64/tutorial_load_save_image.html) tutorial you have already learned how to write a matrix to an image file by using the [cv::imwrite()](https://docs.opencv.org/4.3.0/d4/da8/group__imgcodecs.html#gabbc7ef1aa2edfaa87772f1202d67e0ce) function. However, for debugging purposes it's much more convenient to see the actual values. You can do this using the << operator of _Mat_. Be aware that this only works for two dimensional matrices.

Although _Mat_ works really well as an image container, it is also a general matrix class. Therefore, it is possible to create and manipulate multidimensional matrices. You can create a Mat object in multiple ways:

*   [cv::Mat::Mat](https://docs.opencv.org/4.3.0/d3/d63/classcv_1_1Mat.html#af1d014cecd1510cdf580bf2ed7e5aafc) Constructor

    cout << "M = " << endl << " " << M << endl << endl;

    ![](https://docs.opencv.org/4.3.0/MatBasicContainerOut1.png)

    For two dimensional and multichannel images we first define their size: row and column count wise.

    Then we need to specify the data type to use for storing the elements and the number of channels per matrix point. To do this we have multiple definitions constructed according to the following convention:

    CV_\[The number of bits per item\]\[Signed or Unsigned\]\[Type Prefix\]C\[The channel number\]

    For instance, _CV_8UC3_ means we use unsigned char types that are 8 bit long and each pixel has three of these to form the three channels. There are types predefined for up to four channels. The [cv::Scalar](https://docs.opencv.org/4.3.0/dc/d84/group__core__basic.html#ga599fe92e910c027be274233eccad7beb) is four element short vector. Specify it and you can initialize all matrix points with a custom value. If you need more you can create the type with the upper macro, setting the channel number in parenthesis as you can see below.

*   Use C/C++ arrays and initialize via constructor

    int sz\[3\] = {2,2,2};

    Mat L(3,sz, [CV_8UC](https://docs.opencv.org/4.3.0/d1/d1b/group__core__hal__interface.html#ga78c5506f62d99edd7e83aba259250394)(1), Scalar::all(0));

    The upper example shows how to create a matrix with more than two dimensions. Specify its dimension, then pass a pointer containing the size for each dimension and the rest remains the same.

*   [cv::Mat::create](https://docs.opencv.org/4.3.0/d3/d63/classcv_1_1Mat.html#a55ced2c8d844d683ea9a725c60037ad0) function:

    M.create(4,4, [CV_8UC](https://docs.opencv.org/4.3.0/d1/d1b/group__core__hal__interface.html#ga78c5506f62d99edd7e83aba259250394)(2));

    cout << "M = "<< endl << " " << M << endl << endl;

    ![](https://docs.opencv.org/4.3.0/MatBasicContainerOut2.png)

    You cannot initialize the matrix values with this construction. It will only reallocate its matrix data memory if the new size will not fit into the old one.

*   MATLAB style initializer: [cv::Mat::zeros](https://docs.opencv.org/4.3.0/d3/d63/classcv_1_1Mat.html#a0b57b6a326c8876d944d188a46e0f556) , [cv::Mat::ones](https://docs.opencv.org/4.3.0/d3/d63/classcv_1_1Mat.html#a69ae0402d116fc9c71908d8508dc2f09) , [cv::Mat::eye](https://docs.opencv.org/4.3.0/d3/d63/classcv_1_1Mat.html#a2cf9b9acde7a9852542bbc20ef851ed2) . Specify size and data type to use:

    Mat E = Mat::eye(4, 4, [CV_64F](https://docs.opencv.org/4.3.0/d1/d1b/group__core__hal__interface.html#ga30a562691cc5987bc88eb7bb7a8faf2b));

    cout << "E = " << endl << " " << E << endl << endl;

    Mat O = Mat::ones(2, 2, [CV_32F](https://docs.opencv.org/4.3.0/d1/d1b/group__core__hal__interface.html#ga4a3def5d72b74bed31f5f8ab7676099c));

    cout << "O = " << endl << " " << O << endl << endl;

    Mat Z = Mat::zeros(3,3, [CV_8UC1](https://docs.opencv.org/4.3.0/d1/d1b/group__core__hal__interface.html#ga81df635441b21f532fdace401e04f588));

    cout << "Z = " << endl << " " << Z << endl << endl;

    ![](https://docs.opencv.org/4.3.0/MatBasicContainerOut3.png)

*   For small matrices you may use comma separated initializers or initializer lists (C++11 support is required in the last case):

    Mat C = (Mat_<double>(3,3) << 0, -1, 0, -1, 5, -1, 0, -1, 0);

    cout << "C = " << endl << " " << C << endl << endl;

    C = (Mat_<double>({0, -1, 0, -1, 5, -1, 0, -1, 0})).reshape(3);

    cout << "C = " << endl << " " << C << endl << endl;

    ![](https://docs.opencv.org/4.3.0/MatBasicContainerOut6.png)

*   Create a new header for an existing _Mat_ object and [cv::Mat::clone](https://docs.opencv.org/4.3.0/d3/d63/classcv_1_1Mat.html#adff2ea98da45eae0833e73582dd4a660) or [cv::Mat::copyTo](https://docs.opencv.org/4.3.0/d3/d63/classcv_1_1Mat.html#a33fd5d125b4c302b0c9aa86980791a77) it.

    Mat RowClone = C.row(1).clone();

    cout << "RowClone = " << endl << " " << RowClone << endl << endl;

    ![](https://docs.opencv.org/4.3.0/MatBasicContainerOut7.png)

    Note

    You can fill out a matrix with random values using the [cv::randu()](https://docs.opencv.org/4.3.0/d2/de8/group__core__array.html#ga1ba1026dca0807b27057ba6a49d258c0) function. You need to give a lower and upper limit for the random values:

    [randu](https://docs.opencv.org/4.3.0/d2/de8/group__core__array.html#ga1ba1026dca0807b27057ba6a49d258c0)(R, Scalar::all(0), Scalar::all(255));

    Output formatting
    -----------------


In the above examples you could see the default formatting option. OpenCV, however, allows you to format your matrix output:

*   Default

    cout << "R (default) = " << endl << R << endl << endl;

    ![](https://docs.opencv.org/4.3.0/MatBasicContainerOut8.png)

*   Python

    cout << "R (python) = " << endl << format(R, Formatter::FMT_PYTHON) << endl << endl;

    ![](https://docs.opencv.org/4.3.0/MatBasicContainerOut16.png)

*   Comma separated values (CSV)

    cout << "R (csv) = " << endl << format(R, Formatter::FMT_CSV ) << endl << endl;

    ![](https://docs.opencv.org/4.3.0/MatBasicContainerOut10.png)

*   Numpy

    cout << "R (numpy) = " << endl << format(R, Formatter::FMT_NUMPY ) << endl << endl;

    ![](https://docs.opencv.org/4.3.0/MatBasicContainerOut9.png)

*   C

    cout << "R (c) = " << endl << format(R, Formatter::FMT_C ) << endl << endl;

    ![](https://docs.opencv.org/4.3.0/MatBasicContainerOut11.png)


Output of other common items
----------------------------

OpenCV offers support for output of other common OpenCV data structures too via the << operator:

*   2D Point

    cout << "Point (2D) = " << P << endl << endl;

    ![](https://docs.opencv.org/4.3.0/MatBasicContainerOut12.png)

*   3D Point

    cout << "Point (3D) = " << P3f << endl << endl;

    ![](https://docs.opencv.org/4.3.0/MatBasicContainerOut13.png)

*   std::vector via [cv::Mat](https://docs.opencv.org/4.3.0/d3/d63/classcv_1_1Mat.html "n-dimensional dense array class ")

    vector<float> v;

    v.push_back( (float)[CV_PI](https://docs.opencv.org/4.3.0/db/de0/group__core__utils.html#ga677b89fae9308b340ddaebf0dba8455f)); v.push\_back(2); v.push\_back(3.01f);

    cout << "Vector of floats via Mat = " << Mat(v) << endl << endl;

    ![](https://docs.opencv.org/4.3.0/MatBasicContainerOut14.png)

*   std::vector of points

    vector<Point2f> vPoints(20);

    for (size_t i = 0; i < vPoints.size(); ++i)

    vPoints\[i\] = [Point2f](https://docs.opencv.org/4.3.0/dc/d84/group__core__basic.html#ga7d080aa40de011e4410bca63385ffe2a)((float)(i * 5), (float)(i % 7));

    cout << "A vector of 2D Points = " << vPoints << endl << endl;

    ![](https://docs.opencv.org/4.3.0/MatBasicContainerOut15.png)


Most of the samples here have been included in a small console application. You can download it from [here](https://github.com/opencv/opencv/tree/master/samples/cpp/tutorial_code/core/mat_the_basic_image_container/mat_the_basic_image_container.cpp) or in the core section of the cpp samples.

You can also find a quick video demonstration of this on [YouTube](https://www.youtube.com/watch?v=1tibU7vGWpk).
