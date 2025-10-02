#include "CNN.h"
#include "MNISTConverter.hpp"
#include <arrayfire.h>
#include <iostream>
#include <chrono>
#include <algorithm>
#include <random>

int main() {
    try {
        af::setDevice(0);
        std::cout << "=== GPU 加速 CNN MNIST 训练 ===" << std::endl;
        af::info();
        
        // 加载数据
        MNISTConverter converter;
        std::cout << "\n加载 MNIST 数据集..." << std::endl;
        
        af::array train_images = converter.load_images("../raw/train-images-idx3-ubyte");
        af::array train_labels = converter.load_labels("../raw/train-labels-idx1-ubyte");
        af::array test_images = converter.load_images("../raw/t10k-images-idx3-ubyte");
        af::array test_labels = converter.load_labels("../raw/t10k-labels-idx1-ubyte");
        
        // 创建 CNN
        std::cout << "\n构建 CNN 网络..." << std::endl;
        ConvolutionalNeuralNetwork cnn(train_images, train_labels);
        
        // 添加网络层
        cnn.add_conv_layer(32, 3, 2);  // 32个3x3卷积核，2x2池化
        cnn.add_conv_layer(64, 3, 2);  // 64个3x3卷积核，2x2池化  
        cnn.add_fc_layer(128);         // 128个神经元
        cnn.add_fc_layer(10);          // 10个输出（0-9）
        
        std::cout << "网络总层数: " << cnn.get_num_layers() << std::endl;
        
        // 训练
        std::cout << "\n开始训练..." << std::endl;
        auto start_time = std::chrono::high_resolution_clock::now();
        
        cnn.train(5, 32, 0.001);  // 5轮次，批次32，学习率0.001
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
        std::cout << "训练耗时: " << duration.count() << " 秒" << std::endl;
        
        // 最终测试
        std::cout << "\n最终测试..." << std::endl;
        
        // 分批测试避免内存问题
        int total_test_samples = test_images.dims(0);
        int test_batch_size = 500;
        float total_accuracy = 0.0f;
        int num_batches = 0;
        
        for (int start = 0; start < total_test_samples; start += test_batch_size) {
            int end = std::min(start + test_batch_size, total_test_samples);
            int current_batch_size = end - start;
            
            af::array batch_test_data = af::constant(0.0f, current_batch_size, 1, 28, 28, f32);
            af::array batch_test_labels = af::constant(0.0f, current_batch_size, 1, f32);
            
            for (int i = 0; i < current_batch_size; ++i) {
                int idx = start + i;
                batch_test_data(i, af::span, af::span, af::span) = test_images(idx, af::span, af::span, af::span);
                batch_test_labels(i) = test_labels(idx);
            }
            
            float batch_acc = cnn.accuracy(batch_test_data, batch_test_labels);
            total_accuracy += batch_acc;
            num_batches++;
            
            std::cout << "测试批次 " << (start / test_batch_size + 1) << " 准确率: " << batch_acc << std::endl;
        }
        
        float final_acc = total_accuracy / num_batches;
        std::cout << "最终测试准确率: " << final_acc << std::endl;
        
    } catch (const af::exception& e) {
        std::cerr << "ArrayFire 错误: " << e.what() << std::endl;
        return -1;
    } catch (const std::exception& e) {
        std::cerr << "错误: " << e.what() << std::endl;
        return -1;
    }
    
    return 0;
}