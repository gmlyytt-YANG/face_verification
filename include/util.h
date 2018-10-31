/*************************************************************************
 *
 * Copyright (c) 2018 Administrator. All Rights Reserved
 *
 ************************************************************************/

/*
 * @file util.h
 * @author gmlyytt@outlook.com
 * @date 2018/10/27 16:32:00
 * @brief
 * */
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
#include <sstream>
#include <utility>
#include <vector>

using namespace caffe;
using std::string;

typedef unsigned char BYTE;
typedef long long LONG;

enum ErrorState {
    NULLPTR = 1,
    INVALIDINPUT = 2,
    NORMAL = 3,
    FILEERROR = 4
};

const double eps = 1e-12;

void cv_mat_tobyte(cv::Mat &img, BYTE *&image_buff);

void byte_to_cvmat(BYTE *image_buff, int width, int height, cv::Mat &img, int img_type);

void prepare_img(std::vector<cv::Mat>& crop_imgs, const cv::Mat &img, const int img_width, const int crop_size);

#endif //FACE_VERIFICATION_UTIL_H
