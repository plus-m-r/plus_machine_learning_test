import numpy as np
from torch_network import torch_network
from skopt import gp_minimize
from skopt.space import Real, Integer
from getData import load_mnist_flattened_toNumpy

train_imgs, train_labels, test_imgs, test_labels = load_mnist_flattened_toNumpy()

def objective(params):
    learning_rate, batch_size, hidden_dim1,hidden_dim2 = params
    net = torch_network(train_imgs, train_labels)
    net.add_layer(int(hidden_dim1))
    net.add_layer(int(hidden_dim2))
    net.add_layer(10,True)
    print(f"使用批次:{int(batch_size)}|学习率:{learning_rate}")
    net.fit(epochs=5, batch_size=int(batch_size), learning_rate=learning_rate)
    acc = net.accuracy(test_imgs, test_labels)
    print(f"本次accuracy:{acc}")
    return -acc  # 贝叶斯优化最小化目标

space = [
    Real(1e-4, 1e-1, name='learning_rate', prior='log-uniform'),
    Integer(64, 1024, name='batch_size'),
    Integer(10, 784, name='hidden_dim1'),
    Integer(10, 784, name='hidden_dim2'),
]

res = gp_minimize(objective, space, n_calls=12, random_state=42)
print(f"最优学习率: {res.x[0]}, 最优批次: {int(res.x[1])}, 最优隐藏层1: {int(res.x[2])},最优隐藏层2: {int(res.x[3])},最优准确率: {-res.fun:.4f}")