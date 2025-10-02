#ifndef MNIST_CONVERTER_H
#define MNIST_CONVERTER_H

#include <arrayfire.h>
#include <string>
#include <stdexcept>
#include <fstream>
#include <iostream>
#include <cstdint> 
#include <algorithm>

class MNISTConverter {
private:
    template <typename T>
    T swap_endian(T value) {
        static_assert(std::is_arithmetic<T>::value, "仅支持数值类型");
        union {
            T value;
            char bytes[sizeof(T)];
        } u{};
        u.value = value;
        std::reverse(u.bytes, u.bytes + sizeof(T));
        return u.value;
    }

public:
    af::array load_images(const std::string& file_path) {
        std::ifstream file(file_path, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("无法打开图像文件: " + file_path);
        }

        // 读取 MNIST 头部（4个32位整数：魔数、样本数、行数、列数）
        uint32_t magic, num_images, rows, cols;
        file.read(reinterpret_cast<char*>(&magic), 4);
        file.read(reinterpret_cast<char*>(&num_images), 4);
        file.read(reinterpret_cast<char*>(&rows), 4);
        file.read(reinterpret_cast<char*>(&cols), 4);

        // 转换大端到小端（MNIST 存储为大端）
        magic = swap_endian(magic);
        num_images = swap_endian(num_images);
        rows = swap_endian(rows);
        cols = swap_endian(cols);

        // 验证 MNIST 图像文件（魔数2051，尺寸28×28）
        if (magic != 2051) {
            throw std::runtime_error("不是MNIST图像文件！魔数: " + std::to_string(magic));
        }
        if (rows != 28 || cols != 28) {
            throw std::runtime_error("MNIST图像尺寸错误！应为28×28，实际: " + std::to_string(rows) + "×" + std::to_string(cols));
        }

        // 读取像素数据（uint8，共 num_images × 28 × 28 个像素）
        size_t pixel_count = num_images * rows * cols;
        std::vector<uint8_t> raw_data(pixel_count);
        file.read(reinterpret_cast<char*>(raw_data.data()), pixel_count);
        if (!file) {
            throw std::runtime_error("读取图像数据失败！可能文件损坏");
        }

        // 转换为 float 并归一化到 [0.0, 1.0]
        std::vector<float> normalized_data(pixel_count);
        for (size_t i = 0; i < pixel_count; ++i) {
            normalized_data[i] = static_cast<float>(raw_data[i]) / 255.0f;
        }

        // 关键修复：正确构造 ArrayFire 数组维度 [N, C, H, W]
        // 数据顺序：每个样本是 1×28×28，所以从向量映射到 [N, 1, 28, 28]
        af::array images = af::array(rows, cols, 1, num_images, normalized_data.data(), afHost);
        // 维度重排：从 [H, W, C, N] 转为 [N, C, H, W]（符合 CNN 输入要求）
        images = af::reorder(images, 3, 2, 0, 1);

        // 验证最终维度（必须是 [num_images, 1, 28, 28]）
        af::dim4 final_dims = images.dims();
        std::cout << "成功加载图像: " << num_images << " 张，维度: " 
                  << final_dims[0] << " (样本数) × " 
                  << final_dims[1] << " (通道) × " 
                  << final_dims[2] << " (高度) × " 
                  << final_dims[3] << " (宽度)" << std::endl;

        if (final_dims[2] != 28 || final_dims[3] != 28) {
            throw std::runtime_error("图像维度错误！最终尺寸应为28×28，实际: " + std::to_string(final_dims[2]) + "×" + std::to_string(final_dims[3]));
        }

        return images;
    }

    af::array load_labels(const std::string& file_path) {
        std::ifstream file(file_path, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("无法打开标签文件: " + file_path);
        }

        uint32_t magic, num_labels;
        file.read(reinterpret_cast<char*>(&magic), 4);
        file.read(reinterpret_cast<char*>(&num_labels), 4);

        magic = swap_endian(magic);
        num_labels = swap_endian(num_labels);

        if (magic != 2049) {
            throw std::runtime_error("不是MNIST标签文件！魔数: " + std::to_string(magic));
        }

        std::vector<uint8_t> raw_labels(num_labels);
        file.read(reinterpret_cast<char*>(raw_labels.data()), num_labels);
        if (!file) {
            throw std::runtime_error("读取标签数据失败！可能文件损坏");
        }

        // 构造 [num_labels, 1] 维度的标签数组（便于后续计算）
        std::vector<int> labels(num_labels);
        for (size_t i = 0; i < num_labels; ++i) {
            labels[i] = static_cast<int>(raw_labels[i]);
        }
        af::array labels_af = af::array(num_labels, 1, labels.data(), afHost);

        std::cout << "成功加载标签: " << num_labels << " 个，维度: " << labels_af.dims() << std::endl;
        return labels_af;
    }
};

#endif // MNIST_CONVERTER_H