from CNN import ConvolutionalNeuralNetwork
from getData import load_mnist_with_channel_toNumpy

# 加载MNIST数据（28×28）
train_imgs, train_labels, test_imgs, test_labels = load_mnist_with_channel_toNumpy()

# 初始化网络
network = ConvolutionalNeuralNetwork(train_imgs, train_labels)

# 添加卷积池化层（参数确保卷积输出能被池化窗口整除）
# 计算：卷积输出高度 = (28 - 3 + 2*1) // 1 + 1 = 28，能被2整除
network.add_convolution_pooling_layer(
    output_channels=1,
    kernel_size=3,       # 3×3卷积核
    pool_window_size=2,  # 2×2池化窗口
    stride_size=1,       # 步长1
    padding_size=1       # 填充1
)

# 添加输出层
network.add_fully_connected_layer(10, True)

# 训练网络
network.train_network(
    training_epochs=6,
    number_of_samples_per_batch=32,
    learning_rate=0.001
)

# 测试准确率
test_accuracy = network.calculate_accuracy(test_imgs, test_labels)
print(f"测试准确率: {test_accuracy:.4f}")
    
    