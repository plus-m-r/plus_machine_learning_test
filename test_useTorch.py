from torch_network import torch_network
import numpy as np
from getData import load_mnist_flattened_toNumpy

train_imgs, train_labels, test_imgs, test_labels = load_mnist_flattened_toNumpy()

network = torch_network(train_imgs,train_labels)

network.add_layer(512)
network.add_layer(128)
network.add_layer(10,is_output=True)

network.fit(7,1024,0.0007)

print(f"real accuracy:{network.accuracy(test_imgs,test_labels)}")