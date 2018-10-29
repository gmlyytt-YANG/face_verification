#ifndef FACE_VERIFICATION_FEATURE_OPERATION_H
#define FACE_VERIFICATION_FEATURE_OPERATION_H

#include <util.h>

struct EyePoint {
    int xleft;        //左眼在图像中的横坐标
    int yleft;        //左眼在图像中的纵坐标
    int xright;
    int yright;
    float confidence; //保留;
};

/*
 * @brief 从单幅人脸图像中抽取特征模版（所有特征模板的长度都是相同的）
 * @param pbImageBuff : 输入参数，内存块形式的图像文件
 * @param width       : 输入参数，图片宽度
 * @param height      : 输入参数，输入图片高度
 * @param img_type    : 输入参数，通道类型
 * @param pcTemplateFile : 输入参数，字符串类型，给定保存特征模板的特征文件文件名（含完整路径）
 * @param EyePoint       : 输出参数，结构体指针，给定保存检测到的人眼在图像中的位置信息（输入NULL指针即可，函数内部会自动分配）
 * @param nENum          : 输出参数，眼睛对数，目前为1
 */
extern "C" LONG FRCreateTemplateF(BYTE *pbImageBuff, int width, int height, int img_type, const char *pcTemplateFile,
                                  EyePoint *&psEyePoints, int &nENum);

/*
 * @brief 比较两个输入的特征模板文件，返回相似度
 * @param pcTemplateFileA  : 输入参数，字符串类型，给定参与比对的第1个特征模板文件的文件名（含完整路径）
 * @param pcTemplateFileB  : 输入参数，字符串类型，给定参与比对的第2个特征模板文件的文件名（含完整路径）
 * @param pfSim            : 输出参数，两个特征模板的相似度，float类型，大小为0～1.00
 */
extern "C" LONG FRTemplateMatch(const char *pcTemplateFileA, const char *pcTemplateFileB, float *pfSim);

#endif //FACE_VERIFICATION_FEATURE_OPERATION_H
