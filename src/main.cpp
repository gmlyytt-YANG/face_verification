#include <feature_operation.h>

int main() {
    cv::Mat img = cv::imread("../data/Aaron_Eckhart_0001.jpg", -1);
    const char* file_name = "../data/feature.txt";
    BYTE* img_buff = nullptr;
    cv_mat_tobyte(img, img_buff);
    EyePoint* eye_point = nullptr;
    int eye_num = 0;
    FRCreateTemplateF(img_buff, img.cols, img.rows, img.type(), file_name, eye_point, eye_num);
    return 0;
}
