import torch.nn as nn
import torch
import torch.nn.init as init
import torch.optim as optim
from torch_dataload_process import custom_Dataset
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
class LeNet5(nn.Module):
    def __init__(self,num_classes = 10):
        super(LeNet5,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=6,kernel_size=5,padding=0)
        self.pool1 = nn.AvgPool2d(kernel_size=2,stride=2)
        self.conv2 = nn.Conv2d(in_channels=6,out_channels=16,kernel_size=5,stride=1,padding=0)
        self.pool2 = nn.AvgPool2d(kernel_size=2,stride=2)
        self.conv3 = nn.Conv2d(in_channels=16,out_channels=120,kernel_size=5,stride=1,padding=0)
        self.fc1 = nn.Linear(in_features=120,out_features=84)
        self.fc2 = nn.Linear(in_features=84,out_features=num_classes)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self._initialize_weights()
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                init.normal_(m.weight,mean=0.0,std=0.01)
                if m.bias is not None:
                    init.constant_(m.bias,0.0)
            elif isinstance(m,nn.Linear):
                init.normal_(m.weight,mean=0.0,std=0.01)
                init.constant_(m.bias,1.0)

    def forward(self,X:torch.Tensor) -> torch.Tensor:
        X = self.relu(self.conv1(X))
        X = self.pool1(X)
        X = self.relu(self.conv2(X))
        X = self.pool2(X)
        X = self.relu(self.conv3(X))
        X = X.view(-1,120)
        X = self.relu(self.fc1(X))
        X = self.sigmoid(self.fc2(X))
        return X
    def train_model(self,train_loader,test_loader,epochs=10,lr=0.001):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)
        criterion = nn.BCELoss() if self.fc2.out_features ==1 else nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(),lr = lr)

        train_losses = []
        test_accs=[]

        for epoch in range(epochs):
            self.train()
            total_loss = 0.0
            for images,labels in train_loader:
                images,labels = images.to(device),labels.to(device)
                if isinstance(criterion,nn.BCELoss):
                    labels = torch.nn.functional.one_hot(labels, num_classes=self.fc2.out_features).float()
                outputs = self(images)
                loss = criterion(outputs,labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
            avg_loss = total_loss / len(train_loader)
            train_losses.append(avg_loss)
            test_acc = self.acc(test_loader)
            test_accs.append(test_acc)
            print(f"Epoch [{epoch+1}/{epochs}], 训练损失: {avg_loss:.4f}, 测试准确率: {test_acc:.2f}%")
        
        return train_losses, test_accs
    def acc(self, data_loader):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)
        self.eval()  # 评估模式
        correct = 0
        total = 0
        
        with torch.no_grad():  # 关闭梯度计算，节省内存
            for images, labels in data_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = self(images)
                
                # 取预测概率最大的类别
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        return 100 * correct / total
if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Pad(padding=2,fill=0,padding_mode="constant"),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.1307], std=[0.3081])
    ])
    dataset_train = custom_Dataset("./data/MNIST/raw/",True,transform=transform)
    mnist_train_dataloader = DataLoader(
        dataset=dataset_train,
        batch_size=1024,
        shuffle=True,
        num_workers=4
    )
    dataset_test = custom_Dataset("./data/MNIST/raw/",False,transform=transform)
    mnist_test_dataloader = DataLoader(
        dataset = dataset_test,
        batch_size=1024,
        shuffle=True,
        num_workers=4
    )

    model = LeNet5(10)
    print(model)

    total_params = 0
    for name, param in model.named_parameters():
        param_count = np.prod(param.size())
        total_params += param_count
        print(f"参数名称: {name:20} | 形状: {str(param.shape):20} | 参数数量: {param_count:8} | 数据类型: {param.dtype}")
    
    print("\n" + "="*80 + "\n")
    print(f"模型总参数数量: {total_params}")

    print("\n开始训练...")
    train_losses, test_accs = model.train_model(
        train_loader=mnist_train_dataloader,
        test_loader=mnist_test_dataloader,
        epochs=10,
        lr=0.001
    )

    print(model.acc(mnist_test_dataloader))



