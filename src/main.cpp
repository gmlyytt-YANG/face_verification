#include "feature_operation.h"

int main() {
    cv::Mat img = cv::imread("../data/Aaron_Peirsol_0001.jpg", -1);
    cv::Mat img2 = cv::imread("../data/Aaron_Peirsol_0002.jpg", -1);
    const char* file_name = "../data/feature.txt";
    const char* file_name2 = "../data/feature2.txt";
    BYTE* img_buff = nullptr;
    BYTE* img_buff2 = nullptr;
    cv_mat_tobyte(img, img_buff);
    cv_mat_tobyte(img2, img_buff2);
    EyePoint* eye_point = nullptr;
    int eye_num = 0;
    FRCreateTemplateF(img_buff, img.cols, img.rows, img.type(), file_name, eye_point, eye_num);
    FRCreateTemplateF(img_buff2, img2.cols, img2.rows, img2.type(), file_name2, eye_point, eye_num);
	float sim = 0.0;
	FRTemplateMatch(file_name, file_name2, &sim);
	std::cout << sim << std::endl;
    return 0;
}
