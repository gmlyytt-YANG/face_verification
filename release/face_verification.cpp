#include "face_verification.h"

const double eps = 1e-12;

class Classifier {
public:
    static Classifier *InitClassifier(const string &model_file,
                                      const string &trained_file,
                                      const string &mean_file) {
        if (!instance) {
            instance = new Classifier(model_file, trained_file, mean_file);
        }
        return instance;
    }

    std::vector <std::vector<float>> Predict(const cv::Mat &img);

private:
    Classifier(const string &model_file,
               const string &trained_file,
               const string &mean_file);

    void SetMean(const string &mean_file);

    void WrapInputLayer(std::vector <cv::Mat> *input_channels);

    void Preprocess(const cv::Mat &img,
                    std::vector <cv::Mat> *input_channels);

private:
    static Classifier *instance;
    shared_ptr <Net<float>> net_;
    cv::Size input_geometry_;
    int num_channels_;
    cv::Mat mean_;
    const static int img_width = 256;
    const static int img_height = 256;
    const static int crop_size = 224;
};


Classifier::Classifier(const string &model_file,
                       const string &trained_file,
                       const string &mean_file) {
    Caffe::set_mode(Caffe::GPU);
    /* Load the network. */
    net_.reset(new Net<float>(model_file, TEST));
    net_->CopyTrainedLayersFrom(trained_file);

    CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
    CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";

    Blob<float> *input_layer = net_->input_blobs()[0];
    num_channels_ = input_layer->channels();
    CHECK(num_channels_ == 3 || num_channels_ == 1)
            << "Input layer should have 1 or 3 channels.";
    input_geometry_ = cv::Size(input_layer->width(), input_layer->height());

    /* Load the binaryproto mean file. */
    SetMean(mean_file);
}

/* Load the mean file in binaryproto format. */
void Classifier::SetMean(const string &mean_file) {
    BlobProto blob_proto;
    ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);

    /* Convert from BlobProto to Blob<float> */
    Blob<float> mean_blob;
    mean_blob.FromProto(blob_proto);
    CHECK_EQ(mean_blob.channels(), num_channels_)
            << "Number of channels of mean file doesn't match input layer.";

    /* The format of the mean file is planar 32-bit float BGR or grayscale. */
    std::vector <cv::Mat> channels;
    float *data = mean_blob.mutable_cpu_data();
    for (int i = 0; i < num_channels_; ++i) {
        /* Extract an individual channel. */
        cv::Mat channel(mean_blob.height(), mean_blob.width(), CV_32FC1, data);
        channels.push_back(channel);
        data += mean_blob.height() * mean_blob.width();
    }

    /* Merge the separate channels into a single image. */
    cv::Mat mean;
    cv::merge(channels, mean);

    /* Compute the global mean pixel value and create a mean image
     * filled with this value. */
    cv::Scalar channel_mean = cv::mean(mean);
    mean_ = cv::Mat(input_geometry_, mean.type(), channel_mean);
}

std::vector <vector<float>> Classifier::Predict(const cv::Mat &img) {
    std::vector <cv::Mat> crop_imgs(10);
    std::vector <std::vector<float>> result;
    Blob<float> *input_layer = net_->input_blobs()[0];
    input_layer->Reshape(1, num_channels_,
                         input_geometry_.height, input_geometry_.width);
    /* Forward dimension change to all layers. */
    net_->Reshape();
    prepare_img(crop_imgs, img, img_width, crop_size);
    for (int i = 0; i < crop_imgs.size(); ++i) {
        std::vector <cv::Mat> input_channels;
        WrapInputLayer(&input_channels);
        Preprocess(crop_imgs[i], &input_channels);
        net_->Forward();

        /* Copy the output layer to a std::vector */
        Blob<float> *output_layer = net_->output_blobs()[0];
        const float *begin = output_layer->cpu_data();
        const float *end = begin + output_layer->channels();

        result.push_back(std::vector<float>(begin, end));
    }
    return result;
}

/* Wrap the input layer of the network in separate cv::Mat objects
 * (one per channel). This way we save one memcpy operation and we
 * don't need to rely on cudaMemcpy2D. The last preprocessing
 * operation will write the separate channels directly to the input
 * layer. */
void Classifier::WrapInputLayer(std::vector <cv::Mat> *input_channels) {
    Blob<float> *input_layer = net_->input_blobs()[0];

    int width = input_layer->width();
    int height = input_layer->height();
    float *input_data = input_layer->mutable_cpu_data();
    for (int i = 0; i < input_layer->channels(); ++i) {
        cv::Mat channel(height, width, CV_32FC1, input_data);
        input_channels->push_back(channel);
        input_data += width * height;
    }
}

void Classifier::Preprocess(const cv::Mat &img,
                            std::vector <cv::Mat> *input_channels) {
    /* Convert the input image to the input image format of the network. */
    cv::Mat sample;
    if (img.channels() == 3 && num_channels_ == 1)
        cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
    else if (img.channels() == 4 && num_channels_ == 1)
        cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
    else if (img.channels() == 4 && num_channels_ == 3)
        cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
    else if (img.channels() == 1 && num_channels_ == 3)
        cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
    else
        sample = img;

    cv::Mat sample_resized;
    if (sample.size() != input_geometry_)
        cv::resize(sample, sample_resized, input_geometry_);
    else
        sample_resized = sample;

    cv::Mat sample_float;
    if (num_channels_ == 3)
        sample_resized.convertTo(sample_float, CV_32FC3);
    else
        sample_resized.convertTo(sample_float, CV_32FC1);

    cv::Mat sample_normalized;
    cv::subtract(sample_float, mean_, sample_normalized);

    /* This operation will write the separate BGR planes directly to the
     * input layer of the network because it is wrapped by the cv::Mat
     * objects in input_channels. */
    cv::split(sample_normalized, *input_channels);

    CHECK(reinterpret_cast<float *>(input_channels->at(0).data)
          == net_->input_blobs()[0]->cpu_data())
            << "Input channels are not wrapping the input layer of the network.";
}

Classifier *Classifier::instance = nullptr;


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

void prepare_img(std::vector <cv::Mat> &crop_imgs, const cv::Mat &img, const int img_width, const int crop_size) {
    vector<int> indices = {0, img_width - crop_size};
    int n = 0;
    for (auto &i : indices) {
        for (auto &j : indices) {
            crop_imgs[n] = img.rowRange(i, i + crop_size).colRange(j, j + crop_size);
            cv::flip(crop_imgs[n], crop_imgs[n + 5], 1);
            ++n;
        }
    }
    int center = indices[1] / 2 + 1;
    crop_imgs[4] = img.rowRange(center, center + crop_size).colRange(center, center + crop_size);
    cv::flip(crop_imgs[4], crop_imgs[9], 1);
}


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
    Classifier *classifier = Classifier::InitClassifier(model_file, trained_file, mean_file);
    std::vector <std::vector<float>> feature = classifier->Predict(img);
    for (auto &elem : feature) {
        for (auto &sub_elem : elem) {
            fout << sub_elem << std::endl;
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
        *pfSim = static_cast<float>(feature_numerator /
                                    (sqrt(feature_A_square_sum) * sqrt(feature_B_square_sum) + eps));
        return NORMAL;
    }
}



