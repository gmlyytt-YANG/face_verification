#ifndef FACE_VERIFICATION_FEATURE_OPERATION_H
#define FACE_VERIFICATION_FEATURE_OPERATION_H

#include <util.h>

struct EyePoint {
    int xleft;            //左眼在图像中的横坐标
    int yleft;            //左眼在图像中的纵坐标
    int xright;
    int yright;
    float confidence;  //保留;
};


//功能：读取识别算法的标识（版本等信息）
extern "C" char *FRGetCode();


//释放

//////////////////////////////////////////////////////////////////////////
//功能：从单幅人脸图像中抽取特征模版（所有特征模板的长度都是相同的）---> (7109 B)
//参数：
//     pbImageBuff : 输入参数，内存块形式的图像文件，目前只支持bmp和jpg格式（包含头）
//     nBuffLen    : 输入参数，内存中图像文件的大小，按BYTE算
//     pcTemplateFile :输入参数，字符串类型，给定保存特征模板的特征文件文件名（含完整路径）
//     EyePoint	      :输出参数，结构体指针，给定保存检测到的人眼在图像中的位置信息（输入NULL指针即可，函数内部会自动分配）
//     nENum          :输出参数，眼睛对数，目前为1
extern "C" LONG FRCreateTemplateF(BYTE *pbImageBuff, int width, int height, int img_type, const char *pcTemplateFile,
                                  EyePoint *&psEyePoints, int &nENum);

//////////////////////////////////////////////////////////////////////////
//功能：从单幅人脸图像中抽取特征模版（所有特征模板的长度都是相同的）
//参数：
//     pbImageBuff : 输入参数，内存块形式的图像文件，目前只支持bmp和jpg格式（包含头）
//     nBuffLen    : 输入参数，内存中图像文件的大小，按BYTE算
//     pbTemplateBuff: 输入参数，给定保存特征模板的内存地址（无需事先分配，函数内部会自动分配）
//     EyePoint	      :输出参数，结构体指针，给定保存检测到的人眼在图像中的位置信息（输入NULL指针即可，函数内部会自动分配）
//     nENum          :输出参数，眼睛对数，目前为1
extern "C" LONG
FRCreateTemplateB(BYTE *pbImageBuff, int nBuffLen, BYTE *&pbTemplateBuff, EyePoint *&psEyePoints, int &nENum);

//////////////////////////////////////////////////////////////////////////
//功能：比较两个输入的特征模板文件，返回相似度
//参数：
//     pcTemplateFileA  : 输入参数，字符串类型，给定参与比对的第1个特征模板文件的文件名（含完整路径）
//     pcTemplateFileB  : 输入参数，字符串类型，给定参与比对的第2个特征模板文件的文件名（含完整路径）
//     pfSim            : 输出参数，两个特征模板的相似度，float类型，大小为0～1.00
extern "C" LONG FRTemplateMatch(const char *pcTemplateFileA, const char *pcTemplateFileB, float *pfSim);

//////////////////////////////////////////////////////////////////////////
//功能：比较两个输入的特征模板文件，返回相似度（和上一函数的区别是直接在内存中对已经批量读取好的特征模板进行比对，速度更快）
//参数：
//     pData1           : 输入参数，无符号字符类型，给定参与比对的第1个特征模板的数据(Binary值)
//     pData2           : 输入参数，无符号字符类型，给定参与比对的第2个特征模板的数据(Binary值)
//     nLen1            : 输入参数，pData1数据块的长度
//     nLen2            : 输入参数，pData2数据块的长度
//     pfSim            : 输出参数，两个特征模板的相似度，float类型，大小为0～1.00
extern "C" LONG
FRMemoryMatch(const unsigned char *pData1, const unsigned char *pData2, int nLen1, int nLen2, float *pfSim);


//////////////////////////////////////////////////////////////////////////
#endif //FACE_VERIFICATION_FEATURE_OPERATION_H
