import numpy as np
from torchvision import datasets, transforms
import nltk
from nltk.corpus import ptb

def download_Mnist():
    """下载MNIST数据集到本地（复用原下载逻辑）"""
    datasets.MNIST("data", train=True, download=True, transform=transforms.ToTensor())
    datasets.MNIST("data", train=False, download=True, transform=transforms.ToTensor())

def flatten_images(images):
    """通用图像扁平化工具函数（复用扁平化逻辑）
    
    参数:
        images: 形状为(样本数, 28, 28)的二维图像数组
    返回:
        形状为(样本数, 784)的一维特征数组（784=28×28）
    """
    num_samples = images.shape[0]
    return images.reshape(num_samples, -1)

def load_mnist_toNumpy():
    """加载MNIST数据集，返回原始形状的NumPy数组
    
    返回:
        train_images: 训练集图像，形状(60000, 28, 28)
        train_labels: 训练集标签，形状(60000,)
        test_images: 测试集图像，形状(10000, 28, 28)
        test_labels: 测试集标签，形状(10000,)
    """
    download_Mnist()  # 确保数据已下载
    
    # 加载原始数据集（ToTensor转换后为张量）
    train_set = datasets.MNIST("data", train=True, download=False, transform=transforms.ToTensor())
    test_set = datasets.MNIST("data", train=False, download=False, transform=transforms.ToTensor())
    
    # 转换为NumPy数组并去除灰度图的通道维度（(样本数,1,28,28)→(样本数,28,28)）
    train_images = np.array([img.numpy() for img, _ in train_set]).squeeze()
    train_labels = np.array([label for _, label in train_set])
    test_images = np.array([img.numpy() for img, _ in test_set]).squeeze()
    test_labels = np.array([label for _, label in test_set])
    
    return train_images, train_labels, test_images, test_labels

def load_mnist_flattened_toNumpy():
    """加载MNIST数据集，返回扁平化后的NumPy数组
    
    返回:
        train_flat: 扁平化训练集图像，形状(60000, 784)
        train_labels: 训练集标签，形状(60000,)
        test_flat: 扁平化测试集图像，形状(10000, 784)
        test_labels: 测试集标签，形状(10000,)
    """
    train_imgs, train_labels, test_imgs, test_labels = load_mnist_toNumpy()
    
    # 对原始图像进行扁平化处理
    train_flat = flatten_images(train_imgs)
    test_flat = flatten_images(test_imgs)
    
    return train_flat, train_labels, test_flat, test_labels
def load_mnist_with_channel_toNumpy():
    """加载MNIST数据集，返回包含通道数的NumPy数组
    返回:
        train_images: 训练集图像，形状(60000, 1, 28, 28)
        train_labels: 训练集标签，形状(60000,)
        test_images: 测试集图像，形状(10000, 1, 28, 28)
        test_labels: 测试集标签，形状(10000,)
    """
    download_Mnist()  # 确保数据已下载
    train_set = datasets.MNIST("data", train=True, download=False, transform=transforms.ToTensor())
    test_set = datasets.MNIST("data", train=False, download=False, transform=transforms.ToTensor())
    # 转换为NumPy数组，保留通道维度
    train_images = np.array([img.numpy() for img, _ in train_set])  # (60000, 1, 28, 28)
    train_labels = np.array([label for _, label in train_set])
    test_images = np.array([img.numpy() for img, _ in test_set])    # (10000, 1, 28, 28)
    test_labels = np.array([label for _, label in test_set])
    return train_images, train_labels, test_images, test_labels
def download_PTB():
    nltk.download('ptb')
    sentences = ptb.sents()
    print(len(sentences))
