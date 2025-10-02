cmake_minimum_required(VERSION 3.18)
project(ArrayFireCNN LANGUAGES CXX CUDA)

# --------------------------
# 1. 基础编译配置
# --------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CUDA_ARCHITECTURES 89 CACHE STRING "CUDA 计算架构" FORCE)
set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCHITECTURES})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -march=native")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O2 -lineinfo")

# --------------------------
# 2. ArrayFire 配置
# --------------------------
set(AF_LOCAL_DIR "${CMAKE_SOURCE_DIR}/third_party/arrayfire/ArrayFire-3.9.0-Linux")

if(EXISTS "${AF_LOCAL_DIR}")
    message(STATUS "✅ 使用项目本地 ArrayFire: ${AF_LOCAL_DIR}")
    
    # 设置包含目录和库目录
    set(AF_INCLUDE_DIRS "${AF_LOCAL_DIR}/include")
    set(AF_LIBRARY_DIRS "${AF_LOCAL_DIR}/lib64")
    
    # 查找库文件
    find_library(AF_LIB af PATHS ${AF_LIBRARY_DIRS} NO_DEFAULT_PATH)
    find_library(AFCUDA_LIB afcuda PATHS ${AF_LIBRARY_DIRS} NO_DEFAULT_PATH)
    
    if(AF_LIB AND AFCUDA_LIB)
        set(AF_LIBRARIES ${AFCUDA_LIB} ${AF_LIB})
        message(STATUS "✅ 找到 ArrayFire 库: ${AF_LIBRARIES}")
    else()
        # 尝试查找任何可用的 ArrayFire 库
        file(GLOB AF_LIBS "${AF_LIBRARY_DIRS}/libaf*.so*")
        if(AF_LIBS)
            message(STATUS "✅ 找到 ArrayFire 库文件: ${AF_LIBS}")
            set(AF_LIBRARIES ${AF_LIBS})
        else()
            message(FATAL_ERROR "❌ 在 ${AF_LIBRARY_DIRS} 中未找到 ArrayFire 库文件")
        endif()
    endif()
    
    # 包含目录
    include_directories(${AF_INCLUDE_DIRS})
    link_directories(${AF_LIBRARY_DIRS})
    
else()
    message(FATAL_ERROR "ArrayFire 未找到。请检查安装路径: ${AF_LOCAL_DIR}")
endif()

# --------------------------
# 3. 项目头文件包含目录
# --------------------------
# 添加项目自己的头文件目录
include_directories(${CMAKE_SOURCE_DIR}/include)
message(STATUS "✅ 项目头文件目录: ${CMAKE_SOURCE_DIR}/include")

# --------------------------
# 4. Google Test 配置
# --------------------------
# 添加 gtest 子目录
add_subdirectory(third_party/googletest)

# 包含 gtest 头文件
include_directories(${gtest_SOURCE_DIR}/include)

# --------------------------
# 5. 主程序编译
# --------------------------
file(GLOB MAIN_SOURCES 
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
)

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/main.cpp")
    list(APPEND MAIN_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/main.cpp")
endif()

if(MAIN_SOURCES)
    add_executable(cnn_app ${MAIN_SOURCES})
    target_link_libraries(cnn_app PRIVATE ${AF_LIBRARIES})
    
    # 设置运行时库路径
    set_target_properties(cnn_app PROPERTIES
        BUILD_RPATH "${AF_LIBRARY_DIRS}"
        INSTALL_RPATH "${AF_LIBRARY_DIRS}"
    )
    
    message(STATUS "✅ 主程序目标：cnn_app")
else()
    message(STATUS "ℹ️  未找到主程序源码，仅编译测试目标")
endif()

# --------------------------
# 6. Google Test 测试配置
# --------------------------
enable_testing()

# 创建测试可执行文件
add_executable(arrayfire_tests
    test/test_arrayfire_complete.cpp
)

# 链接库
target_link_libraries(arrayfire_tests
    PRIVATE ${AF_LIBRARIES}
    PRIVATE gtest gtest_main
)

target_compile_features(arrayfire_tests PRIVATE cxx_std_17)

# 设置运行时库路径
set_target_properties(arrayfire_tests PROPERTIES
    BUILD_RPATH "${AF_LIBRARY_DIRS}"
    INSTALL_RPATH "${AF_LIBRARY_DIRS}"
)

# 注册测试
add_test(NAME ArrayFire_Complete_Tests COMMAND arrayfire_tests)

# --------------------------
# 7. 项目信息
# --------------------------
message(STATUS "----------------------------------------")
message(STATUS "【ArrayFire CNN 项目配置完成】")
message(STATUS "ArrayFire 路径: ${AF_LOCAL_DIR}")
message(STATUS "ArrayFire 库: ${AF_LIBRARIES}")
message(STATUS "项目头文件目录: ${CMAKE_SOURCE_DIR}/include")
message(STATUS "----------------------------------------")
message(STATUS "【测试命令说明】")
message(STATUS "在 build 目录运行：")
message(STATUS "  运行所有测试：./arrayfire_tests")
message(STATUS "  运行特定测试：./arrayfire_tests --gtest_filter=TestClass.TestName")
message(STATUS "  列出所有测试：./arrayfire_tests --gtest_list_tests")
message(STATUS "  详细输出：    ./arrayfire_tests --gtest_verbose")
message(STATUS "  主程序演示：  ./cnn_app")
message(STATUS "----------------------------------------")