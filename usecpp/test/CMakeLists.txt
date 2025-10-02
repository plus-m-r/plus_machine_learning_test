cmake_minimum_required(VERSION 3.18)
project(ArrayFireTests LANGUAGES CXX CUDA)

# 基础配置
if(DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES})
endif()

# ArrayFire 配置
set(AF_LOCAL_DIR "${CMAKE_SOURCE_DIR}/third_party/arrayfire/ArrayFire-3.9.0-Linux")

if(EXISTS "${AF_LOCAL_DIR}")
    message(STATUS "✅ 测试使用 ArrayFire: ${AF_LOCAL_DIR}")
    
    # 设置包含目录和库目录
    set(AF_INCLUDE_DIRS "${AF_LOCAL_DIR}/include")
    set(AF_LIBRARY_DIRS "${AF_LOCAL_DIR}/lib64")
    
    # 查找库文件
    find_library(AF_LIB af PATHS ${AF_LIBRARY_DIRS} NO_DEFAULT_PATH)
    find_library(AFCUDA_LIB afcuda PATHS ${AF_LIBRARY_DIRS} NO_DEFAULT_PATH)
    
    if(AF_LIB AND AFCUDA_LIB)
        set(AF_LIBRARIES ${AFCUDA_LIB} ${AF_LIB})
        message(STATUS "✅ 找到 ArrayFire 测试库: ${AF_LIBRARIES}")
    else()
        # 尝试查找任何可用的 ArrayFire 库
        file(GLOB AF_LIBS "${AF_LIBRARY_DIRS}/libaf*.so*")
        if(AF_LIBS)
            message(STATUS "✅ 找到 ArrayFire 测试库文件: ${AF_LIBS}")
            set(AF_LIBRARIES ${AF_LIBS})
        else()
            message(FATAL_ERROR "❌ 在 ${AF_LIBRARY_DIRS} 中未找到 ArrayFire 测试库文件")
        endif()
    endif()
    
    # 包含目录
    include_directories(${AF_INCLUDE_DIRS})
    link_directories(${AF_LIBRARY_DIRS})
    
else()
    message(FATAL_ERROR "ArrayFire 未找到。请检查安装路径: ${AF_LOCAL_DIR}")
endif()

# 创建测试可执行文件
add_executable(arrayfire_tests
    test_arrayfire_complete.cpp
)

# 链接库
target_link_libraries(arrayfire_tests
    PRIVATE ${AF_LIBRARIES}
    PRIVATE gtest gtest_main
)

# 设置编译标准
target_compile_features(arrayfire_tests PRIVATE cxx_std_17)

# 设置运行时库路径
set_target_properties(arrayfire_tests PROPERTIES
    BUILD_RPATH "${AF_LIBRARY_DIRS}"
    INSTALL_RPATH "${AF_LIBRARY_DIRS}"
)

message(STATUS "✅ ArrayFire 测试目标创建完成: arrayfire_tests")