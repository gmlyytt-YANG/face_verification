#ifndef FACE_VERIFICATION_UTIL_H
#define FACE_VERIFICATION_UTIL_H

#include <iostream>
#include <caffe/caffe.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <algorithm>
#include <fstream>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>

using namespace caffe;
using std::string;

typedef unsigned char BYTE;
typedef long long LONG;

enum ErrorState{
    NULLPTR = 1,
    INVALIDINPUT = 2,
    NORMAL = 3,
    FILEERROR = 4
};

void cv_mat_tobyte(cv::Mat &img, BYTE *&image_buff);

void byte_to_cvmat(BYTE *image_buff, int width, int height, cv::Mat &img, int img_type);

#endif //FACE_VERIFICATION_UTIL_H
