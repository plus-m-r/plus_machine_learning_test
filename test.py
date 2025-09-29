import numpy as np
from getData import load_mnist_flattened_toNumpy
from network import Network

train_imgs, train_labels, test_imgs, test_labels = load_mnist_flattened_toNumpy()

network = Network(train_imgs,train_labels)

network.add_layer(784,392)
network.add_layer(392,196)
network.add_layer(196,98)
network.add_layer(98,49)
network.add_layer(49,10,True)

network.train(7,1256,0.02)

print(f"准确率{network.accuracy(test_imgs,test_labels)}")
