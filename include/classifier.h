/*************************************************************************
 *
 * Copyright (c) 2018 Administrator. All Rights Reserved
 *
 ************************************************************************/

/*
 * @file classifier.h
 * @author gmlyytt@outlook.com
 * @date 2018/10/26 16:32:00
 * @brief
 * */
#ifndef _CLASSIFIER_H
#define _CLASSIFIER_H

#include "../include/util.h"

class Classifier {
public:
    static Classifier* InitClassifier(const string &model_file, 
                               const string &trained_file,
                               const string &mean_file) {
        if (!instance) {
            instance = new Classifier(model_file, trained_file, mean_file);
        }
        return instance;
    }

    std::vector<std::vector<float>> Predict(const cv::Mat &img);

private:
    Classifier(const string &model_file,
               const string &trained_file,
               const string &mean_file);

    void SetMean(const string &mean_file);

    void WrapInputLayer(std::vector<cv::Mat> *input_channels);

    void Preprocess(const cv::Mat &img,
                    std::vector<cv::Mat> *input_channels);

private:
    static Classifier* instance;
    shared_ptr<Net<float> > net_;
    cv::Size input_geometry_;
    int num_channels_;
    cv::Mat mean_;
    const static int img_width = 256;
    const static int img_height = 256;
    const static int crop_size = 224;
};

#endif
