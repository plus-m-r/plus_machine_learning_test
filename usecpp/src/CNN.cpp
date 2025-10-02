#include "CNN.h"
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <cfloat>
#include <iostream>

// 1. 构造函数
ConvolutionalNeuralNetwork::ConvolutionalNeuralNetwork(const af::array& data, const af::array& labels) 
    : training_data(data), training_labels(labels), num_training_samples(data.dims(0)) {
    
    std::cout << "初始化 CNN - 输入维度: " << data.dims() << std::endl;
}

// 2. ReLU 激活函数
af::array ConvolutionalNeuralNetwork::relu(const af::array& x) {
    return af::max(0.0f, x);
}

// 3. Softmax 激活函数
af::array ConvolutionalNeuralNetwork::softmax(const af::array& x) {
    af::array max_vals = af::max(x, 1);
    af::array exp_vals = af::exp(x - af::tile(max_vals, 1, x.dims(1)));
    af::array sum_vals = af::sum(exp_vals, 1);
    return exp_vals / af::tile(sum_vals, 1, x.dims(1));
}

// 4. 卷积操作
af::array ConvolutionalNeuralNetwork::conv2d(const af::array& input, const af::array& weights, 
                                           const af::array& biases, int stride, int padding) {
    
    try {
        af::dim4 input_dims = input.dims();
        af::dim4 weight_dims = weights.dims();
        
        int batch_size = input_dims[0];
        int in_channels = input_dims[1];
        int height = input_dims[2];
        int width = input_dims[3];
        
        int out_channels = weight_dims[0];
        int kernel_size = weight_dims[2];
        
        // 计算输出尺寸
        int out_height = (height - kernel_size + 2 * padding) / stride + 1;
        int out_width = (width - kernel_size + 2 * padding) / stride + 1;
        
        // 创建输出数组 [batch_size, out_channels, out_height, out_width]
        af::array output = af::constant(0.0f, batch_size, out_channels, out_height, out_width, f32);
        
        // 将数据复制到CPU进行手动卷积
        float* input_host = input.host<float>();
        float* weights_host = weights.host<float>();
        float* biases_host = biases.host<float>();
        float* output_host = new float[batch_size * out_channels * out_height * out_width];
        
        // 手动实现卷积
        for (int b = 0; b < batch_size; ++b) {
            for (int oc = 0; oc < out_channels; ++oc) {
                for (int oh = 0; oh < out_height; ++oh) {
                    for (int ow = 0; ow < out_width; ++ow) {
                        float sum = 0.0f;
                        
                        for (int ic = 0; ic < in_channels; ++ic) {
                            for (int kh = 0; kh < kernel_size; ++kh) {
                                for (int kw = 0; kw < kernel_size; ++kw) {
                                    int h_index = oh * stride - padding + kh;
                                    int w_index = ow * stride - padding + kw;
                                    
                                    // 边界检查
                                    if (h_index >= 0 && h_index < height && w_index >= 0 && w_index < width) {
                                        int input_idx = b * in_channels * height * width + 
                                                       ic * height * width + 
                                                       h_index * width + 
                                                       w_index;
                                        
                                        int weight_idx = oc * in_channels * kernel_size * kernel_size + 
                                                        ic * kernel_size * kernel_size + 
                                                        kh * kernel_size + 
                                                        kw;
                                        
                                        sum += input_host[input_idx] * weights_host[weight_idx];
                                    }
                                }
                            }
                        }
                        
                        // 添加偏置
                        sum += biases_host[oc];
                        
                        int output_idx = b * out_channels * out_height * out_width + 
                                        oc * out_height * out_width + 
                                        oh * out_width + 
                                        ow;
                        output_host[output_idx] = sum;
                    }
                }
            }
        }
        
        // 将结果写回GPU
        output.write(output_host, batch_size * out_channels * out_height * out_width * sizeof(float), afHost);
        
        // 释放内存
        af::freeHost(input_host);
        af::freeHost(weights_host);
        af::freeHost(biases_host);
        delete[] output_host;
        
        return output;

    } catch (const af::exception& e) {
        std::cerr << "卷积错误（ArrayFire）: " << e.what() << std::endl;
        throw;
    } catch (const std::exception& e) {
        std::cerr << "卷积错误（逻辑）: " << e.what() << std::endl;
        throw;
    }
}

// 5. 最大池化
af::array ConvolutionalNeuralNetwork::max_pool2d(const af::array& input, int pool_size) {
    try {
        af::dim4 input_dims = input.dims();
        
        if (input_dims.ndims() != 4) {
            throw std::runtime_error("池化输入必须是4维数组 [N, C, H, W]！实际维度数: " + std::to_string(input_dims.ndims()));
        }
        
        int N = input_dims[0];
        int C = input_dims[1];
        int H = input_dims[2];
        int W = input_dims[3];
        
        int out_H = H / pool_size;
        int out_W = W / pool_size;
        
        if (out_H * pool_size != H || out_W * pool_size != W) {
            throw std::runtime_error("输入尺寸不能被池化尺寸整除！输入: " + std::to_string(H) + "×" + std::to_string(W) 
                                   + "，池化尺寸: " + std::to_string(pool_size));
        }
        
        // 手动实现池化
        size_t output_elements = N * C * out_H * out_W;
        af::array output = af::constant(0.0f, N, C, out_H, out_W, f32);
        
        float* input_host = input.host<float>();
        float* output_host = new float[output_elements];
        
        for (int n = 0; n < N; ++n) {
            for (int c = 0; c < C; ++c) {
                for (int oh = 0; oh < out_H; ++oh) {
                    for (int ow = 0; ow < out_W; ++ow) {
                        int h_start = oh * pool_size;
                        int w_start = ow * pool_size;
                        int h_end = h_start + pool_size;
                        int w_end = w_start + pool_size;
                        
                        float max_val = -FLT_MAX;
                        for (int h = h_start; h < h_end; ++h) {
                            for (int w = w_start; w < w_end; ++w) {
                                int index = n * C * H * W + c * H * W + h * W + w;
                                if (input_host[index] > max_val) {
                                    max_val = input_host[index];
                                }
                            }
                        }
                        
                        int out_index = n * C * out_H * out_W + c * out_H * out_W + oh * out_W + ow;
                        output_host[out_index] = max_val;
                    }
                }
            }
        }
        
        output.write(output_host, output_elements * sizeof(float), afHost);
        
        af::freeHost(input_host);
        delete[] output_host;
        
        return output;

    } catch (const af::exception& e) {
        std::cerr << "池化错误（ArrayFire）: " << e.what() << std::endl;
        throw;
    } catch (const std::exception& e) {
        std::cerr << "池化错误（逻辑）: " << e.what() << std::endl;
        throw;
    }
}

// 6. 添加卷积层
void ConvolutionalNeuralNetwork::add_conv_layer(int output_channels, int kernel_size, int pool_size) {
    int input_channels = conv_weights.empty() ? training_data.dims(1) : conv_output_channels.back();
    
    // He 初始化
    float scale = std::sqrt(2.0f / (input_channels * kernel_size * kernel_size));
    af::array weights = af::randn(output_channels, input_channels, kernel_size, kernel_size, f32) * scale;
    af::array biases = af::constant(0.0f, output_channels, 1, f32);
    
    conv_weights.push_back(weights);
    conv_biases.push_back(biases);
    conv_output_channels.push_back(output_channels);
    conv_kernel_sizes.push_back(kernel_size);
    pool_sizes.push_back(pool_size);
    
    // 初始化梯度
    conv_weight_grads.push_back(af::constant(0.0f, output_channels, input_channels, kernel_size, kernel_size, f32));
    conv_bias_grads.push_back(af::constant(0.0f, output_channels, 1, f32));
    
    std::cout << "添加卷积层: " << input_channels << " -> " << output_channels 
              << " 核:" << kernel_size << " 池化:" << pool_size << std::endl;
}

// 7. 添加全连接层
void ConvolutionalNeuralNetwork::add_fc_layer(int output_dim) {
    int input_dim;
    
    if (fc_weights.empty()) {
        if (!conv_weights.empty()) {
            int last_conv_channels = conv_output_channels.back();
            input_dim = last_conv_channels * 7 * 7;
        } else {
            input_dim = training_data.dims(1) * training_data.dims(2) * training_data.dims(3);
        }
    } else {
        input_dim = fc_weights.back().dims(1);
    }
    
    // He 初始化
    float scale = std::sqrt(2.0f / input_dim);
    af::array weights = af::randn(input_dim, output_dim, f32) * scale;
    af::array biases = af::constant(0.0f, 1, output_dim, f32);
    
    fc_weights.push_back(weights);
    fc_biases.push_back(biases);
    
    // 初始化梯度
    fc_weight_grads.push_back(af::constant(0.0f, input_dim, output_dim, f32));
    fc_bias_grads.push_back(af::constant(0.0f, 1, output_dim, f32));
    
    std::cout << "添加全连接层: " << input_dim << " -> " << output_dim << std::endl;
}

// 8. 前向传播
std::pair<af::array, std::vector<af::array>> ConvolutionalNeuralNetwork::forward(const af::array& input) {
    std::vector<af::array> activations;
    af::array current = input;
    
    // 保存输入
    activations.push_back(current);
    
    // 卷积层前向传播
    for (size_t i = 0; i < conv_weights.size(); ++i) {
        // 卷积 + ReLU
        current = conv2d(current, conv_weights[i], conv_biases[i], 1, 1);
        activations.push_back(current);  // 保存卷积输出
        
        current = relu(current);
        activations.push_back(current);  // 保存ReLU输出
        
        // 池化
        current = max_pool2d(current, pool_sizes[i]);
        activations.push_back(current);  // 保存池化输出
    }
    
    // 展平
    if (current.numdims() > 2) {
        int batch_size = current.dims(0);
        int total_elements = current.elements() / batch_size;
        current = af::moddims(current, batch_size, total_elements);
    }
    activations.push_back(current);  // 保存展平输出
    
    // 全连接层前向传播
    for (size_t i = 0; i < fc_weights.size(); ++i) {
        current = af::matmul(current, fc_weights[i]) + fc_biases[i];
        activations.push_back(current);  // 保存线性输出
        
        if (i < fc_weights.size() - 1) {
            current = relu(current);
            activations.push_back(current);  // 保存ReLU输出
        } else {
            current = softmax(current);
            activations.push_back(current);  // 保存softmax输出
        }
    }
    
    return {current, activations};
}

// 9. 交叉熵损失
float ConvolutionalNeuralNetwork::cross_entropy_loss(const af::array& predictions, const af::array& labels) {
    int n = predictions.dims(0);
    af::array selected_probs = af::constant(0.0f, n, f32);
    
    for (int i = 0; i < n; ++i) {
        int label_idx = static_cast<int>(labels(i).scalar<float>());
        selected_probs(i) = predictions(i, label_idx).scalar<float>();
    }
    
    af::array log_probs = af::log(selected_probs + 1e-10f);
    return -af::mean<float>(log_probs);
}

// 10. 反向传播 - 简化版本
void ConvolutionalNeuralNetwork::backward(const af::array& input, const af::array& labels, 
                                        const std::vector<af::array>& activations) {
    
    int batch_size = input.dims(0);
    
    // 计算输出层的梯度 (softmax + 交叉熵的梯度简化)
    af::array predictions = activations.back();
    af::array d_output = predictions;
    
    // 将真实标签转换为one-hot编码并减去
    for (int i = 0; i < batch_size; ++i) {
        int true_label = static_cast<int>(labels(i).scalar<float>());
        d_output(i, true_label) = d_output(i, true_label) - 1.0f;
    }
    d_output = d_output / batch_size;
    
    // 全连接层反向传播
    int activation_idx = activations.size() - 2;  // softmax前的线性输出
    
    for (int i = fc_weights.size() - 1; i >= 0; --i) {
        // 计算权重梯度
        af::array layer_input = (i == 0) ? activations[activations.size() - 2 * fc_weights.size() - 1] 
                                        : activations[activation_idx - 1];
        fc_weight_grads[i] = af::matmul(layer_input.T(), d_output);
        
        // 计算偏置梯度
        fc_bias_grads[i] = af::sum(d_output, 0);
        
        // 计算传递到前一层的梯度
        if (i > 0) {
            d_output = af::matmul(d_output, fc_weights[i].T());
            
            // ReLU梯度
            af::array relu_output = activations[activation_idx - 1];
            d_output = d_output * (relu_output > 0.0f);
            
            activation_idx -= 2;  // 移动到前一个线性层
        }
    }
    
    // 注意：这里简化了卷积层的反向传播，实际应该更复杂
    // 为了简化演示，我们只更新全连接层
}

// 11. 参数更新
void ConvolutionalNeuralNetwork::update_parameters(float learning_rate) {
    // 更新全连接层参数
    for (size_t i = 0; i < fc_weights.size(); ++i) {
        fc_weights[i] = fc_weights[i] - learning_rate * fc_weight_grads[i];
        fc_biases[i] = fc_biases[i] - learning_rate * fc_bias_grads[i];
    }
    
    // 注意：这里简化了，实际应该也更新卷积层参数
}

// 12. 训练函数
void ConvolutionalNeuralNetwork::train(int epochs, int batch_size, float learning_rate) {
    if (epochs <= 0 || batch_size <= 0) {
        throw std::invalid_argument("训练参数必须为正数");
    }
    
    int n = num_training_samples;
    batch_size = std::min(batch_size, n);
    
    std::cout << "\n开始训练: " << epochs << " 轮次, 批次大小 " << batch_size 
              << ", 学习率 " << learning_rate << std::endl;

    // 训练前测试
    std::cout << "\n训练前测试..." << std::endl;
    int test_sample_size = std::min(100, n);
    
    af::array test_data = af::constant(0.0f, test_sample_size, 1, 28, 28, f32);
    af::array test_labels_subset = af::constant(0.0f, test_sample_size, 1, f32);
    
    for (int i = 0; i < test_sample_size; ++i) {
        test_data(i, af::span, af::span, af::span) = training_data(i, af::span, af::span, af::span);
        test_labels_subset(i) = training_labels(i);
    }
    
    // 执行前向传播测试
    try {
        auto [test_preds, _] = forward(test_data);
        std::cout << "训练前测试通过，前向传播输出维度: " << test_preds.dims() << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "训练前测试失败: " << e.what() << std::endl;
        throw;
    }
    
    // 正式训练循环
    for (int epoch = 0; epoch < epochs; ++epoch) {
        float total_loss = 0.0f;
        int num_batches = 0;
        
        // 简单数据打乱
        std::vector<int> indices(n);
        for (int i = 0; i < n; ++i) indices[i] = i;
        std::random_shuffle(indices.begin(), indices.end());
        
        // 批次训练
        for (int start = 0; start < n; start += batch_size) {
            int end = std::min(start + batch_size, n);
            int current_batch_size = end - start;
            
            // 提取批次数据
            af::array batch_data = af::constant(0.0f, current_batch_size, 1, 28, 28, f32);
            af::array batch_labels = af::constant(0.0f, current_batch_size, 1, f32);
            
            for (int i = 0; i < current_batch_size; ++i) {
                int idx = indices[start + i];
                batch_data(i, af::span, af::span, af::span) = training_data(idx, af::span, af::span, af::span);
                batch_labels(i) = training_labels(idx);
            }
            
            try {
                // 前向传播
                auto [predictions, activations] = forward(batch_data);
                
                // 计算损失
                float loss = cross_entropy_loss(predictions, batch_labels);
                total_loss += loss;
                
                // 反向传播
                backward(batch_data, batch_labels, activations);
                
                // 参数更新
                update_parameters(learning_rate);
                
                num_batches++;
                
            } catch (const std::exception& e) {
                std::cerr << "批次 " << (start / batch_size) << " 训练错误: " << e.what() << std::endl;
                continue;
            }
        }
        
        if (num_batches == 0) continue;
        
        // 计算准确率
        int sample_size = std::min(500, n);
        af::array sample_data = af::constant(0.0f, sample_size, 1, 28, 28, f32);
        af::array sample_labels = af::constant(0.0f, sample_size, 1, f32);
        
        for (int i = 0; i < sample_size; ++i) {
            sample_data(i, af::span, af::span, af::span) = training_data(i, af::span, af::span, af::span);
            sample_labels(i) = training_labels(i);
        }
        
        try {
            auto [sample_preds, _] = forward(sample_data);
            af::array max_vals, predicted_classes;
            af::max(max_vals, predicted_classes, sample_preds, 1);
            af::array sample_labels_flat = af::flat(sample_labels);
            float accuracy = af::mean<float>(predicted_classes == sample_labels_flat);
            
            std::cout << "第 " << (epoch + 1) << "/" << epochs << " 轮 | "
                      << "平均损失: " << (total_loss / num_batches) << " | "
                      << "训练准确率: " << accuracy << std::endl;
                      
        } catch (const std::exception& e) {
            std::cout << "第 " << (epoch + 1) << "/" << epochs << " 轮 | "
                      << "平均损失: " << (total_loss / num_batches) << " | "
                      << "准确率评估失败: " << e.what() << std::endl;
        }
    }
}

// 13. 预测函数
af::array ConvolutionalNeuralNetwork::predict(const af::array& input) {
    auto [predictions, _] = forward(input);
    af::array max_vals, predicted_classes;
    af::max(max_vals, predicted_classes, predictions, 1);
    return predicted_classes;
}

// 14. 准确率计算
float ConvolutionalNeuralNetwork::accuracy(const af::array& input, const af::array& labels) {
    af::array predictions = predict(input);
    af::array labels_flat = af::flat(labels);
    return af::mean<float>(predictions == labels_flat);
}