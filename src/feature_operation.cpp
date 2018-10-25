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
    Classifier classifier(model_file, trained_file, mean_file);
    std::vector<float> feature = classifier.Predict(img);
    for (auto& elem : feature) {
        fout << elem << " ";
    }
    fout << std::endl;
    fout.close();
    return NORMAL;
}