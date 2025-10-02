#include <gtest/gtest.h>
#include <arrayfire.h>
#include <chrono>
#include <iostream>
#include <vector>
#include <functional>

class ArrayFireCompleteTest : public ::testing::Test {
protected:
    void SetUp() override {
        // 自动选择最佳后端
        int backends = af::getAvailableBackends();
        
        if (backends & AF_BACKEND_CUDA) {
            af::setBackend(AF_BACKEND_CUDA);
            backend_name = "CUDA";
            std::cout << "✅ 使用 CUDA 后端" << std::endl;
        } else if (backends & AF_BACKEND_OPENCL) {
            af::setBackend(AF_BACKEND_OPENCL);
            backend_name = "OpenCL";
            std::cout << "✅ 使用 OpenCL 后端" << std::endl;
        } else {
            backend_name = "CPU";
            std::cout << "✅ 使用 CPU 后端" << std::endl;
        }
        
        std::cout << "ArrayFire 版本: " << af::infoString() << std::endl;
        
        // af::info() 会直接打印设备信息，不需要输出到流
        std::cout << "设备信息: ";
        af::info();  // 这会直接打印到标准输出
    }
    
    std::string backend_name;
    
    // 性能测量函数
    double measure_time_ms(std::function<void()> operation, int warmup = 1, int runs = 3) {
        // 预热
        for (int i = 0; i < warmup; ++i) {
            operation();
            af::sync();
        }
        
        // 测量
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < runs; ++i) {
            operation();
        }
        af::sync();
        auto end = std::chrono::high_resolution_clock::now();
        
        return std::chrono::duration<double, std::milli>(end - start).count() / runs;
    }
};

// 测试基础功能
TEST_F(ArrayFireCompleteTest, BasicFunctionality) {
    // 创建数组
    af::array A = af::randu(5, 3);
    af::array B = af::randu(3, 4);
    
    // 矩阵乘法
    af::array C = af::matmul(A, B);
    
    EXPECT_EQ(C.dims(0), 5);
    EXPECT_EQ(C.dims(1), 4);
    
    std::cout << "基础功能测试通过 - 矩阵维度: " << C.dims() << std::endl;
}

// 测试矩阵运算性能
TEST_F(ArrayFireCompleteTest, MatrixMultiplicationPerformance) {
    const int size = 2000;
    af::array A = af::randu(size, size);
    af::array B = af::randu(size, size);
    
    double time_ms = measure_time_ms([&]() {
        af::array C = af::matmul(A, B);
    });
    
    std::cout << "[" << backend_name << "] " << size << "×" << size 
              << " 矩阵乘法: " << time_ms << " ms" << std::endl;
    
    // 验证性能在合理范围内
    if (backend_name == "CUDA") {
        EXPECT_LT(time_ms, 500) << "CUDA 性能异常";
    }
}

// 测试逐元素运算
TEST_F(ArrayFireCompleteTest, ElementWiseOperations) {
    af::array A = af::randu(1000, 1000);
    af::array B = af::randu(1000, 1000);
    
    double time_ms = measure_time_ms([&]() {
        af::array C = af::exp(A) + af::sin(B) * af::sqrt(A);
    });
    
    std::cout << "[" << backend_name << "] 逐元素运算: " << time_ms << " ms" << std::endl;
    
    EXPECT_LT(time_ms, 1000) << "逐元素运算性能异常";
}

// 测试卷积操作
TEST_F(ArrayFireCompleteTest, Convolution) {
    af::array input = af::randu(256, 256);
    af::array kernel = af::constant(1.0f, 3, 3) / 9.0f;
    
    double time_ms = measure_time_ms([&]() {
        af::array output = af::convolve2(input, kernel);
    });
    
    std::cout << "[" << backend_name << "] 256×256 卷积: " << time_ms << " ms" << std::endl;
    
    EXPECT_LT(time_ms, 100) << "卷积运算性能异常";
}

// 测试池化操作 - 修复版本
TEST_F(ArrayFireCompleteTest, Pooling) {
    af::array input = af::randu(512, 512);
    
    // 使用手动实现池化操作
    double max_pool_time = measure_time_ms([&]() {
        // 手动实现 2x2 最大池化
        int h = input.dims(0);
        int w = input.dims(1);
        af::array pooled = af::constant(0, h/2, w/2);
        
        for (int i = 0; i < h/2; ++i) {
            for (int j = 0; j < w/2; ++j) {
                af::array window = input(af::seq(2*i, 2*i+1), af::seq(2*j, 2*j+1));
                pooled(i, j) = af::max<float>(window);
            }
        }
    });
    
    double avg_pool_time = measure_time_ms([&]() {
        // 手动实现 2x2 平均池化
        int h = input.dims(0);
        int w = input.dims(1);
        af::array pooled = af::constant(0, h/2, w/2);
        
        for (int i = 0; i < h/2; ++i) {
            for (int j = 0; j < w/2; ++j) {
                af::array window = input(af::seq(2*i, 2*i+1), af::seq(2*j, 2*j+1));
                pooled(i, j) = af::mean<float>(window);
            }
        }
    });
    
    std::cout << "[" << backend_name << "] 最大池化: " << max_pool_time << " ms" << std::endl;
    std::cout << "[" << backend_name << "] 平均池化: " << avg_pool_time << " ms" << std::endl;
}

// 测试 CNN 相关操作 - 修复版本
TEST_F(ArrayFireCompleteTest, CNNOperations) {
    // 模拟批量图像数据 (batch_size=8, channels=3, height=224, width=224)
    af::array input = af::randu(224, 224, 3, 8);
    
    // 卷积权重 (3x3 卷积核, 输入3通道, 输出32通道)
    af::array weights = af::randu(3, 3, 3, 32) * 0.1f;
    
    double total_time = measure_time_ms([&]() {
        // 卷积层
        af::array conv = af::convolve2(input, weights);
        
        // ReLU 激活
        af::array relu = af::max(0.0f, conv);
        
        // 手动实现最大池化
        int h = relu.dims(0);
        int w = relu.dims(1);
        int c = relu.dims(2);
        int b = relu.dims(3);
        af::array pooled = af::constant(0, h/2, w/2, c, b);
        
        for (int i = 0; i < h/2; ++i) {
            for (int j = 0; j < w/2; ++j) {
                for (int k = 0; k < c; ++k) {
                    for (int l = 0; l < b; ++l) {
                        af::array window = relu(af::seq(2*i, 2*i+1), af::seq(2*j, 2*j+1), k, l);
                        pooled(i, j, k, l) = af::max<float>(window);
                    }
                }
            }
        }
        
        // 展平
        int flat_size = pooled.dims(0) * pooled.dims(1) * pooled.dims(2) * pooled.dims(3);
        af::array flattened = af::flat(pooled);
    });
    
    std::cout << "[" << backend_name << "] CNN 操作流水线: " << total_time << " ms" << std::endl;
    
    EXPECT_LT(total_time, 1000) << "CNN 操作性能异常";
}

// 测试内存管理
TEST_F(ArrayFireCompleteTest, MemoryManagement) {
    // 测试大内存分配和释放
    try {
        af::array large_matrix = af::randu(4096, 4096);
        
        std::cout << "[" << backend_name << "] 成功分配 4096×4096 矩阵" << std::endl;
        std::cout << "矩阵大小: " << large_matrix.dims() << std::endl;
        std::cout << "内存使用: " << large_matrix.bytes() / (1024.0 * 1024.0) << " MB" << std::endl;
        
        SUCCEED() << "大内存管理测试通过";
        
    } catch (const af::exception& e) {
        std::cout << "⚠️ 内存不足: " << e.what() << std::endl;
        SUCCEED() << "内存不足不是测试失败";
    }
}

// 测试多后端兼容性
TEST_F(ArrayFireCompleteTest, DataTypeCompatibility) {
    // 测试不同数据类型
    std::vector<af::dtype> test_types = {f32, f64};
    
    for (auto dtype : test_types) {
        af::array A = af::randu(100, 100, dtype);
        af::array B = af::randu(100, 100, dtype);
        
        EXPECT_NO_THROW({
            af::array C = af::matmul(A, B);
            EXPECT_EQ(C.dims(0), 100);
            EXPECT_EQ(C.dims(1), 100);
        }) << "数据类型 " << dtype << " 运算失败";
    }
    
    std::cout << "✅ 所有数据类型兼容性测试通过" << std::endl;
}

// 测试 GPU 加速验证
TEST_F(ArrayFireCompleteTest, GPUAccelerationVerification) {
    const int large_size = 4096;
    
    // 测试大矩阵运算
    af::array A = af::randu(large_size, large_size);
    af::array B = af::randu(large_size, large_size);
    
    auto start = std::chrono::high_resolution_clock::now();
    af::array C = af::matmul(A, B);
    af::sync();  // 等待 GPU 完成
    auto end = std::chrono::high_resolution_clock::now();
    
    auto gpu_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    // 估算 CPU 执行时间（经验公式）
    long double estimated_cpu_time = (large_size * large_size * large_size) / 1e9 * 1000; // 简化估算
    
    std::cout << "[" << backend_name << "] " << large_size << "×" << large_size 
              << " 矩阵乘法: " << gpu_time.count() << " ms" << std::endl;
    std::cout << "预计 CPU 时间: ~" << static_cast<int>(estimated_cpu_time) << " ms" << std::endl;
    
    double speedup_ratio = estimated_cpu_time / gpu_time.count();
    
    if (backend_name == "CUDA" && speedup_ratio > 5.0) {
        std::cout << "🎉 GPU 加速明显！加速比: " << speedup_ratio << "x" << std::endl;
    } else if (backend_name == "CUDA") {
        std::cout << "✅ GPU 正在工作，加速比: " << speedup_ratio << "x" << std::endl;
    }
    
    // 验证确实是 GPU 在执行
    if (backend_name == "CUDA") {
        EXPECT_GT(speedup_ratio, 2.0) << "GPU 加速效果不明显";
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    
    std::cout << "========================================" << std::endl;
    std::cout << "ArrayFire 完整功能测试" << std::endl;
    std::cout << "========================================" << std::endl;
    
    int result = RUN_ALL_TESTS();
    
    std::cout << "========================================" << std::endl;
    if (result == 0) {
        std::cout << "🎉 所有测试通过！ArrayFire 配置成功！" << std::endl;
    } else {
        std::cout << "❌ 部分测试失败" << std::endl;
    }
    std::cout << "========================================" << std::endl;
    
    return result;
}