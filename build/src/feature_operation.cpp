#include "classifier.h"
#include "feature_operation.h"
#include "util.h"

extern "C" LONG
FRCreateTemplateF(BYTE *pbImageBuff, int width, int height, int img_type, const char *pcTemplateFile,
                  EyePoint *&psEyePoints, int &nENum) {
    if (!pbImageBuff) {
        return NULLPTR;
    }
    if (height <= 0 || width <= 0) {
        return INVALIDINPUT;
    }
    std::ofstream fout(pcTemplateFile);
    if (!fout) {
        return FILEERROR;
    }
    cv::Mat img;
    byte_to_cvmat(pbImageBuff, width, height, img, img_type);
    string model_file = "../config/deploy.prototxt";
    string trained_file = "../config/_iter_48800.caffemodel";
    string mean_file = "../config/face_data_mean.binaryproto";
    Classifier* classifier = Classifier::InitClassifier(model_file, trained_file, mean_file);
    std::vector<std::vector<float>> feature = classifier->Predict(img);
    for (auto& elem : feature) {
		for (auto& sub_elem : elem) {
        	fout << sub_elem << std::endl;
		}
    }
    fout.close();
    return NORMAL;
}

extern "C" LONG FRTemplateMatch(const char *pcTemplateFileA, const char *pcTemplateFileB, float *pfSim) {
    if (!pcTemplateFileA || !pcTemplateFileB) {
        return NULLPTR;
    }
    std::ifstream fin_A(pcTemplateFileA);
    std::ifstream fin_B(pcTemplateFileB);
    vector<float> feature_A;
    vector<float> feature_B;
    float read_elem = 0.0;
    double feature_A_square_sum = 0.0;
    double feature_B_square_sum = 0.0;
    double feature_numerator = 0.0;
    std::string str;
    while (fin_A >> str) {
        std::stringstream is;
        is << str;
        is >> read_elem;
        feature_A.push_back(read_elem);
        feature_A_square_sum += std::pow(read_elem, 2);
    }
    while (fin_B >> str) {
        std::stringstream is;
        is << str;
        is >> read_elem;
        feature_B.push_back(read_elem);
        feature_B_square_sum += std::pow(read_elem, 2);
        if (feature_B.size() > feature_A.size()) {
            return INVALIDINPUT;
        }
    }
    if (feature_A.size() != feature_B.size()) {
        return INVALIDINPUT;
    }
    for (int i = 0; i < feature_A.size(); ++i) {
        feature_numerator += feature_A[i] * feature_B[i];
    }
    *pfSim = static_cast<float>(feature_numerator / (sqrt(feature_A_square_sum) * sqrt(feature_B_square_sum) + eps));
    return NORMAL;
}
