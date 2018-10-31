#include "util.h"

void cv_mat_tobyte(cv::Mat &img, BYTE *&image_buff) {
    if (image_buff) {
        delete[] image_buff;
    }
    int width = img.cols;
    int height = img.rows;
    int n_bytes = height * width * img.channels();
    image_buff = new BYTE[n_bytes];
    memcpy(image_buff, img.data, n_bytes);
}

void byte_to_cvmat(BYTE *image_buff, int width, int height, cv::Mat &img, int img_type) {
    if (!image_buff) {
        return;
    }
    if (!img.empty()) {
        img.release();
    }
    int channels = img_type == CV_8UC1 ? 1 : 3;
    int n_bytes = width * height * channels;
    img = cv::Mat::zeros(height, width, img_type);
    memcpy(img.data, image_buff, n_bytes);
}

void prepare_img(std::vector<cv::Mat>& crop_imgs, const cv::Mat &img, const int img_width, const int crop_size) {
    vector<int> indices = {0, img_width - crop_size}; 
    int n = 0;  
    for (auto& i : indices) {
        for (auto& j : indices) { 
            crop_imgs[n] = img.rowRange(i, i + crop_size).colRange(j, j + crop_size); 
            cv::flip(crop_imgs[n], crop_imgs[n + 5], 1); 
			++n;  
        } 
    }   
    int center = indices[1] / 2 + 1;
    crop_imgs[4] = img.rowRange(center, center + crop_size).colRange(center, center + crop_size);
    cv::flip(crop_imgs[4], crop_imgs[9], 1); 
}

