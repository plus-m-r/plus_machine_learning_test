#ifndef CNN_H
#define CNN_H

#include <arrayfire.h>
#include <vector>
#include <string>

class ConvolutionalNeuralNetwork {
private:
    af::array training_data;
    af::array training_labels;
    int num_training_samples;
    
    // 网络参数
    std::vector<af::array> conv_weights;
    std::vector<af::array> conv_biases;
    std::vector<af::array> fc_weights;
    std::vector<af::array> fc_biases;
    
    // 梯度
    std::vector<af::array> conv_weight_grads;
    std::vector<af::array> conv_bias_grads;
    std::vector<af::array> fc_weight_grads;
    std::vector<af::array> fc_bias_grads;
    
    std::vector<int> conv_output_channels;
    std::vector<int> conv_kernel_sizes;
    std::vector<int> pool_sizes;
    
    // 激活函数
    af::array relu(const af::array& x);
    af::array softmax(const af::array& x);
    
    // 网络层
    af::array conv2d(const af::array& input, const af::array& weights, 
                    const af::array& biases, int stride = 1, int padding = 1);
    af::array max_pool2d(const af::array& input, int pool_size);
    
    // 损失函数
    float cross_entropy_loss(const af::array& predictions, const af::array& labels);
    
    // 反向传播
    void backward(const af::array& input, const af::array& labels, 
                 const std::vector<af::array>& activations);
    
    // 参数更新
    void update_parameters(float learning_rate);

public:
    ConvolutionalNeuralNetwork(const af::array& data, const af::array& labels);
    
    // 网络构建
    void add_conv_layer(int output_channels, int kernel_size = 3, int pool_size = 2);
    void add_fc_layer(int output_dim);
    
    // 训练和预测
    std::pair<af::array, std::vector<af::array>> forward(const af::array& input);
    void train(int epochs, int batch_size, float learning_rate);
    af::array predict(const af::array& input);
    float accuracy(const af::array& input, const af::array& labels);
    
    // 工具函数
    int get_num_layers() const {
        return conv_weights.size() + fc_weights.size();
    }
};

#endif // CNN_H