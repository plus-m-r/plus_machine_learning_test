import torch
import torch.nn as nn
import torch.optim as optim

class torch_network(nn.Module):
    def __init__(self, X_train, Y_train):
        """
        初始化神经网络
        参数:
            X_train: 原始训练数据，形状(样本数, 特征数)（需提前扁平化）
            Y_train: 原始训练标签，形状(样本数,)（整数类别，如MNIST的0-9）
        """  
        super().__init__()
        self.X_train = X_train
        self.Y_train = Y_train
        self.layers = nn.ModuleList()
        self.num_train_samples = X_train.shape[1]
    def add_layer(self, output_dim, is_output=False):
        if len(self.layers) == 0:
            input_dim = self.num_train_samples
        else:
            for layer in reversed(self.layers):
                if isinstance(layer, nn.Linear):
                    input_dim = layer.out_features
                    break
        layer = nn.Linear(input_dim,output_dim)
        self.layers.append(layer)
        if not is_output:
            self.layers.append(nn.ReLU())
        print(f"新增层: 输入维度{input_dim} → 输出维度{output_dim}（{'输出层' if is_output else '隐藏层'}）")
    def forward(self,X):
        for layer in self.layers:
            X = layer(X)
        X = nn.functional.softmax(X,dim=1)
        return X
    def fit(self,epochs,batch_size,learning_rate):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        X_train = torch.tensor(self.X_train,dtype = torch.float32).to(device)
        Y_train = torch.tensor(self.Y_train,dtype = torch.long).to(device)
        optimizer = optim.Adam(self.parameters(),lr = learning_rate)
        criterion = nn.CrossEntropyLoss()
        dataset = torch.utils.data.TensorDataset(X_train,Y_train)
        loader = torch.utils.data.DataLoader(dataset,batch_size=batch_size,shuffle=True)
        for epoch in range(epochs):
            total_loss = 0
            correct = 0
            total = 0
            for xb,yb in loader:
                optimizer.zero_grad()
                out = self.forward(xb)
                loss = criterion(out,yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                preds = torch.argmax(out,dim = 1)
                correct += (preds == yb).sum()
                total += yb.size(0)
            avg_loss = total_loss / len(loader)
            acc = correct / total
            print(f"Epoch {epoch+1}/{epochs} | 平均损失: {avg_loss:.4f}| 训练准确率: {acc:.4f}")
    def accuracy(self, X, Y):
        self.eval()
        device = next(self.parameters()).device
        X = torch.tensor(X, dtype=torch.float32).to(device)
        Y = torch.tensor(Y, dtype=torch.long).to(device)
        with torch.no_grad():
            outputs = self.forward(X)
            preds = torch.argmax(outputs, dim=1)
            correct = (preds == Y).sum().item()
            total = Y.size(0)
            return correct / total



    