**Transition guide**
> **NOTE**
> 从 2.x 迁移到 3.0 的 guide，官方文档已经 4.x，然而教程还在 opencv 2.x ……

Changes overview
================

This document is intended to software developers who want to migrate their code to OpenCV 3.0.

OpenCV 3.0 introduced many new algorithms and features comparing to version 2.4. Some modules have been rewritten, some have been reorganized. Although most of the algorithms from 2.4 are still present, the interfaces can differ.

This section describes most notable changes in general, all details and examples of transition actions are in the next part of the document.

##### Contrib repository

[https://github.com/opencv/opencv_contrib](https://github.com/opencv/opencv_contrib)

This is a place for all new, experimental and non-free algorithms. It does not receive so much attention from the support team comparing to main repository, but the community makes an effort to keep it in a good shape.

To build OpenCV with _contrib_ repository, add the following option to your cmake command:

-DOPENCV\_EXTRA\_MODULES\_PATH=<path-to-opencv\_contrib>/modules

##### Headers layout

In 2.4 all headers are located in corresponding module subfolder (_opencv2/<module>/<module>.hpp_), in 3.0 there are top-level module headers containing the most of the module functionality: _opencv2/<module>.hpp_ and all C-style API definitions have been moved to separate headers (for example opencv2/core/core_c.h).

##### Algorithm interfaces

General algorithm usage pattern has changed: now it must be created on heap wrapped in smart pointer [cv::Ptr](https://docs.opencv.org/4.3.0/dc/d84/group__core__basic.html#ga6395ca871a678020c4a31fadf7e8cc63). Version 2.4 allowed both stack and heap allocations, directly or via smart pointer.

_get_ and _set_ methods have been removed from the [cv::Algorithm](https://docs.opencv.org/4.3.0/d3/d46/classcv_1_1Algorithm.html "This is a base class for all more or less complex algorithms in OpenCV. ") class along with _CV\_INIT\_ALGORITHM_ macro. In 3.0 all properties have been converted to the pairs of _getProperty/setProperty_ pure virtual methods. As a result it is **not** possible to create and use [cv::Algorithm](https://docs.opencv.org/4.3.0/d3/d46/classcv_1_1Algorithm.html "This is a base class for all more or less complex algorithms in OpenCV. ") instance by name (using generic _Algorithm::create(String)_ method), one should call corresponding factory method explicitly.

##### Changed modules

*   _ml_ module has been rewritten
*   _highgui_ module has been split into parts: _imgcodecs_, _videoio_ and _highgui_ itself
*   _features2d_ module have been reorganized (some feature detectors has been moved to _opencv_contrib/xfeatures2d_ module)
*   _legacy_, _nonfree_ modules have been removed. Some algorithms have been moved to different locations and some have been completely rewritten or removed
*   CUDA API has been updated (_gpu_ module -> several _cuda_ modules, namespace _gpu_ -\> namespace _cuda_)
*   OpenCL API has changed (_ocl_ module has been removed, separate _ocl::_ implementations -> Transparent API)
*   Some other methods and classes have been relocated

Transition hints
================

This section describes concrete actions with examples.

Prepare 2.4
-----------

Some changes made in the latest 2.4.11 OpenCV version allow you to prepare current codebase to migration:

*   [cv::makePtr](https://docs.opencv.org/4.3.0/dc/d84/group__core__basic.html#gaee940caae29d5569aa3aa9ba77fb887f) function is now available
*   _opencv2/<module>.hpp_ headers have been created

New headers layout
------------------

**Note:** Changes intended to ease the migration have been made in OpenCV 3.0, thus the following instructions are not necessary, but recommended.

1.  Replace inclusions of old module headers:

    #include "opencv2/<module>/<module>.hpp"

    // new header

    #include "opencv2/<module>.hpp"


Modern way to use algorithm
---------------------------

1.  Algorithm instances must be created with [cv::makePtr](https://docs.opencv.org/4.3.0/dc/d84/group__core__basic.html#gaee940caae29d5569aa3aa9ba77fb887f) function or corresponding static factory method if available:

    Ptr<SomeAlgo> algo = makePtr<SomeAlgo>(...);

    Ptr<SomeAlgo> algo = SomeAlgo::create(...);

    Other ways are deprecated:

    Ptr<SomeAlgo> algo = new SomeAlgo(...);

    SomeAlgo * algo = new SomeAlgo(...);

    SomeAlgo algo(...);

    Ptr<SomeAlgo> algo = Algorithm::create<SomeAlgo>("name");

2.  Algorithm properties should be accessed via corresponding virtual methods, _getSomeProperty/setSomeProperty_, generic _get/set_ methods have been removed:

    double clipLimit = clahe->getClipLimit();

    clahe->setClipLimit(clipLimit);

    double clipLimit = clahe->getDouble("clipLimit");

    clahe->set("clipLimit", clipLimit);

    clahe->setDouble("clipLimit", clipLimit);

3.  Remove `initModule_<moduleName>()` calls

Machine learning module
-----------------------

Since this module has been rewritten, it will take some effort to adapt your software to it. All algorithms are located in separate _ml_ namespace along with their base class _StatModel_. Separate _SomeAlgoParams_ classes have been replaced with a sets of corresponding _getProperty/setProperty_ methods.

The following table illustrates correspondence between 2.4 and 3.0 machine learning classes.

| 2.4 | 3.0 |
| --- | --- |
| CvStatModel | [cv::ml::StatModel](https://docs.opencv.org/4.3.0/db/d7d/classcv_1_1ml_1_1StatModel.html "Base class for statistical models in OpenCV ML. ") |
| CvNormalBayesClassifier | [cv::ml::NormalBayesClassifier](https://docs.opencv.org/4.3.0/d4/d8e/classcv_1_1ml_1_1NormalBayesClassifier.html "Bayes classifier for normally distributed data. ") |
| CvKNearest | [cv::ml::KNearest](https://docs.opencv.org/4.3.0/dd/de1/classcv_1_1ml_1_1KNearest.html "The class implements K-Nearest Neighbors model. ") |
| CvSVM | [cv::ml::SVM](https://docs.opencv.org/4.3.0/d1/d2d/classcv_1_1ml_1_1SVM.html "Support Vector Machines. ") |
| CvDTree | [cv::ml::DTrees](https://docs.opencv.org/4.3.0/d8/d89/classcv_1_1ml_1_1DTrees.html "The class represents a single decision tree or a collection of decision trees. ") |
| CvBoost | [cv::ml::Boost](https://docs.opencv.org/4.3.0/d6/d7a/classcv_1_1ml_1_1Boost.html "Boosted tree classifier derived from DTrees. ") |
| CvGBTrees | _Not implemented_ |
| CvRTrees | [cv::ml::RTrees](https://docs.opencv.org/4.3.0/d0/d65/classcv_1_1ml_1_1RTrees.html "The class implements the random forest predictor. ") |
| CvERTrees | _Not implemented_ |
| EM | [cv::ml::EM](https://docs.opencv.org/4.3.0/d1/dfb/classcv_1_1ml_1_1EM.html "The class implements the Expectation Maximization algorithm. ") |
| CvANN_MLP | [cv::ml::ANN_MLP](https://docs.opencv.org/4.3.0/d0/dce/classcv_1_1ml_1_1ANN__MLP.html "Artificial Neural Networks - Multi-Layer Perceptrons. ") |
| _Not implemented_ | [cv::ml::LogisticRegression](https://docs.opencv.org/4.3.0/d6/df9/classcv_1_1ml_1_1LogisticRegression.html "Implements Logistic Regression classifier. ") |
| CvMLData | [cv::ml::TrainData](https://docs.opencv.org/4.3.0/dc/d32/classcv_1_1ml_1_1TrainData.html "Class encapsulating training data. ") |

Although rewritten _ml_ algorithms in 3.0 allow you to load old trained models from _xml/yml_ file, deviations in prediction process are possible.

The following code snippets from the `points_classifier.cpp` example illustrate differences in model training process:

using namespace [cv](https://docs.opencv.org/4.3.0/d2/d75/namespacecv.html);

[Mat](https://docs.opencv.org/4.3.0/d3/d63/classcv_1_1Mat.html) trainSamples, trainClasses;

prepare\_train\_data( trainSamples, trainClasses );

CvBoost boost;

var_types.at<[uchar](https://docs.opencv.org/4.3.0/d1/d1b/group__core__hal__interface.html#ga65f85814a8290f9797005d3b28e7e5fc)>( trainSamples.[cols](https://docs.opencv.org/4.3.0/d3/d63/classcv_1_1Mat.html#aa3e5a47585c9ef6a0842556739155e3e) ) = CV\_VAR\_CATEGORICAL;

CvBoostParams params( CvBoost::DISCRETE,

100,

0.95,

2,

false,

0

);

boost.train( trainSamples, CV\_ROW\_SAMPLE, trainClasses, [Mat](https://docs.opencv.org/4.3.0/d3/d63/classcv_1_1Mat.html)(), [Mat](https://docs.opencv.org/4.3.0/d3/d63/classcv_1_1Mat.html)(), var_types, [Mat](https://docs.opencv.org/4.3.0/d3/d63/classcv_1_1Mat.html)(), params );

boost->setBoostType(Boost::DISCRETE);

boost->setWeakCount(100);

boost->setWeightTrimRate(0.95);

boost->setMaxDepth(2);

boost->setUseSurrogates(false);

boost->setPriors([Mat](https://docs.opencv.org/4.3.0/d3/d63/classcv_1_1Mat.html)());

boost->train(prepare\_train\_data());

Features detect
---------------

Some algorithms (FREAK, BRIEF, SIFT, SURF) has been moved to _opencv_contrib_ repository, to _xfeatures2d_ module, _xfeatures2d_ namespace. Their interface has been also changed (inherit from `[cv::Feature2D](https://docs.opencv.org/4.3.0/d0/d13/classcv_1_1Feature2D.html "Abstract base class for 2D image feature detectors and descriptor extractors. ")` base class).

List of _xfeatures2d_ module classes:

*   [cv::xfeatures2d::BriefDescriptorExtractor](https://docs.opencv.org/4.3.0/d1/d93/classcv_1_1xfeatures2d_1_1BriefDescriptorExtractor.html "Class for computing BRIEF descriptors described in  . ") \- Class for computing BRIEF descriptors (2.4 location: _features2d_)
*   [cv::xfeatures2d::FREAK](https://docs.opencv.org/4.3.0/df/db4/classcv_1_1xfeatures2d_1_1FREAK.html "Class implementing the FREAK (Fast Retina Keypoint) keypoint descriptor, described in  ...") \- Class implementing the FREAK (Fast Retina Keypoint) keypoint descriptor (2.4 location: _features2d_)
*   [cv::xfeatures2d::StarDetector](https://docs.opencv.org/4.3.0/dd/d39/classcv_1_1xfeatures2d_1_1StarDetector.html "The class implements the keypoint detector introduced by , synonym of StarDetector. : ") \- The class implements the CenSurE detector (2.4 location: _features2d_)
*   [cv::xfeatures2d::SIFT](https://docs.opencv.org/4.3.0/d5/d3c/classcv_1_1xfeatures2d_1_1SIFT.html "Class for extracting keypoints and computing descriptors using the Scale Invariant Feature Transform ...") \- Class for extracting keypoints and computing descriptors using the Scale Invariant Feature Transform (SIFT) algorithm (2.4 location: _nonfree_)
*   [cv::xfeatures2d::SURF](https://docs.opencv.org/4.3.0/d5/df7/classcv_1_1xfeatures2d_1_1SURF.html "Class for extracting Speeded Up Robust Features from an image  . ") \- Class for extracting Speeded Up Robust Features from an image (2.4 location: _nonfree_)

Following steps are needed:

1.  Add _opencv_contrib_ to compilation process
2.  Include `opencv2/xfeatures2d.h` header
3.  Use namespace `xfeatures2d`
4.  Replace `operator()` calls with `detect`, `compute` or `detectAndCompute` if needed

Some classes now use general methods `detect`, `compute` or `detectAndCompute` provided by `Feature2D` base class instead of custom `operator()`

Following code snippets illustrate the difference (from `video_homography.cpp` example):

using namespace [cv](https://docs.opencv.org/4.3.0/d2/d75/namespacecv.html);

BriefDescriptorExtractor brief(32);

GridAdaptedFeatureDetector detector(new [FastFeatureDetector](https://docs.opencv.org/4.3.0/df/d74/classcv_1_1FastFeatureDetector.html)(10, true), DESIRED_FTRS, 4, 4);

detector.detect(gray, query_kpts);

brief.compute(gray, query\_kpts, query\_desc);

detector->detect(gray, query_kpts);

brief->compute(gray, query\_kpts, query\_desc);

OpenCL
------

All specialized `ocl` implementations has been hidden behind general C++ algorithm interface. Now the function execution path can be selected dynamically at runtime: CPU or OpenCL; this mechanism is also called "Transparent API".

New class [cv::UMat](https://docs.opencv.org/4.3.0/d7/d45/classcv_1_1UMat.html) is intended to hide data exchange with OpenCL device in a convenient way.

Following example illustrate API modifications (from [OpenCV site](http://opencv.org/platforms/opencl.html)):

*   OpenCL-aware code OpenCV-2.x

    VideoCapture vcap(...);

    ocl::OclCascadeClassifier fd("haar_ff.xml");

    ocl::oclMat frame, frameGray;

    Mat frameCpu;

    vector<Rect> faces;

    for(;;){

    vcap >> frameCpu;

    frame = frameCpu;

    fd.detectMultiScale(frameGray, faces, ...);

    }

*   OpenCL-aware code OpenCV-3.x

    VideoCapture vcap(...);

    CascadeClassifier fd("haar_ff.xml");

    UMat frame, frameGray;

    vector<Rect> faces;

    for(;;){

    vcap >> frame;

    [cvtColor](https://docs.opencv.org/4.3.0/d8/d01/group__imgproc__color__conversions.html#ga397ae87e1288a81d2363b61574eb8cab)(frame, frameGray, BGR2GRAY);

    fd.detectMultiScale(frameGray, faces, ...);

    }


CUDA
----

CUDA modules has been moved into opencv_contrib repository.

*   _cuda_ \- [CUDA-accelerated Computer Vision](https://docs.opencv.org/4.3.0/d1/d1e/group__cuda.html)
*   _cudaarithm_ \- [Operations on Matrices](https://docs.opencv.org/4.3.0/d5/d8e/group__cudaarithm.html)
*   _cudabgsegm_ \- [Background Segmentation](https://docs.opencv.org/4.3.0/d6/d17/group__cudabgsegm.html)
*   _cudacodec_ \- [Video Encoding/Decoding](https://docs.opencv.org/4.3.0/d0/d61/group__cudacodec.html)
*   _cudafeatures2d_ \- [Feature Detection and Description](https://docs.opencv.org/4.3.0/d6/d1d/group__cudafeatures2d.html)
*   _cudafilters_ \- [Image Filtering](https://docs.opencv.org/4.3.0/dc/d66/group__cudafilters.html)
*   _cudaimgproc_ \- [Image Processing](https://docs.opencv.org/4.3.0/d0/d05/group__cudaimgproc.html)
*   _cudalegacy_ \- [Legacy support](https://docs.opencv.org/4.3.0/d5/dc3/group__cudalegacy.html)
*   _cudaoptflow_ \- [Optical Flow](https://docs.opencv.org/4.3.0/d7/d3f/group__cudaoptflow.html)
*   _cudastereo_ \- [Stereo Correspondence](https://docs.opencv.org/4.3.0/dd/d47/group__cudastereo.html)
*   _cudawarping_ \- [Image Warping](https://docs.opencv.org/4.3.0/db/d29/group__cudawarping.html)
*   _cudev_ \- [Device layer](https://docs.opencv.org/4.3.0/df/dfc/group__cudev.html)

Documentation format
--------------------

Documentation has been converted to Doxygen format. You can find updated documentation writing guide in _Tutorials_ section of _OpenCV_ reference documentation ([Writing documentation for OpenCV](https://docs.opencv.org/4.3.0/d4/db1/tutorial_documentation.html)).

Support both versions
---------------------

In some cases it is possible to support both versions of OpenCV.

### Source code

To check library major version in your application source code, the following method should be used:

#if CV\_MAJOR\_VERSION == 2

#elif CV\_MAJOR\_VERSION == 3

#endif

Note

Do not use **CV\_VERSION\_MAJOR**, it has different meaning for 2.4 and 3.x branches!

### Build system

It is possible to link different modules or enable/disable some of the features in your application by checking library version in the build system. Standard cmake or pkg-config variables can be used for this:

*   `OpenCV_VERSION` for cmake will contain full version: "2.4.11" or "3.0.0" for example
*   `OpenCV_VERSION_MAJOR` for cmake will contain only major version number: 2 or 3
*   pkg-config file has standard field `Version`

Example:

if(OpenCV\_VERSION VERSION\_LESS "3.0")

\# use 2.4 modules

else()

\# use 3.x modules

endif()
