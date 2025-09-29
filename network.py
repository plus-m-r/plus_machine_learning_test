import numpy as np

def he_initialization(input_dim, output_dim):
    """He初始化（适用于ReLU激活的隐藏层）"""
    if input_dim <= 0 or output_dim <= 0:
        raise ValueError("输入/输出维度必须为正整数")
    std = np.sqrt(2.0 / input_dim)
    return np.random.randn(input_dim, output_dim) * std

class Network:
    def __init__(self, X_train, Y_train):
        """
        初始化神经网络（仅存储原始训练数据，不修改）
        参数:
            X_train: 原始训练数据，形状(样本数, 特征数)（需提前扁平化）
            Y_train: 原始训练标签，形状(样本数,)（整数类别，如MNIST的0-9）
        """  
        self.X_train = X_train
        self.Y_train = Y_train
        self.layers = [] 
        self.num_train_samples = X_train.shape[0]

    def add_layer(self, input_dim, output_dim, is_output=False):
        """
        添加网络层
        参数:
            input_dim: 输入特征维度
            output_dim: 输出特征维度（隐藏层为神经元数，输出层为类别数）
            is_output: 是否为输出层（输出层用Softmax，隐藏层用ReLU）
        """
        if input_dim <= 0 or output_dim <= 0:
            raise ValueError("输入/输出维度必须为正整数")
        W = he_initialization(input_dim, output_dim)
        b = np.zeros((1, output_dim)) 
        
        self.layers.append({
            "W": W,
            "b": b,
            "is_output": is_output
        })
        print(f"新增层: 输入维度{input_dim} → 输出维度{output_dim}（{'输出层' if is_output else '隐藏层'}）")

    def forward(self, X):
        """
        修复索引越界问题：用外部变量跟踪前一层激活值，而非依赖caches索引
        参数:
            X: 输入数据，形状(批次大小, 特征数)
        返回:
            y_pred: 预测概率，形状(批次大小, 类别数)
            caches: 缓存中间结果（Z, A, A_prev），用于反向传播
        """
        caches = []
        A_prev = X  # A_prev：前一层的激活值，初始为输入X
    
        for layer in self.layers:
            W = layer["W"]
            b = layer["b"]
        
            # 1. 线性计算：Z = A_prev · W + b
            Z = np.dot(A_prev, W) + b
        
            # 2. 激活函数
            if layer["is_output"]:
                # Softmax（数值稳定版）
                max_z = np.max(Z, axis=1, keepdims=True)
                exp_Z = np.exp(Z - max_z)
                sum_exp_Z = np.sum(exp_Z, axis=1, keepdims=True)
                sum_exp_Z = np.maximum(sum_exp_Z, 1e-10)  # 防止除0
                current_A = exp_Z / sum_exp_Z
            else:
                # ReLU激活
                current_A = np.maximum(0, Z)
        
            # 3. 缓存当前层的关键信息（A_prev是前一层激活值，直接取自外部变量）
            caches.append({
                "Z": Z,
                "A": current_A,
                "A_prev": A_prev  # 直接存储前一层的A，无需后续计算索引
            })
        
            # 4. 更新A_prev：当前层的A作为下一层的前向激活值
            A_prev = current_A
    
        return current_A, caches  # current_A即y_pred

    def cross_entropy_loss(self, Y, y_pred):
        """
        计算交叉熵损失（接收批次标签和预测概率，不依赖类属性）
        参数:
            Y: 真实标签，形状(批次大小,)（整数类别）
            y_pred: 预测概率，形状(批次大小, 类别数)
        返回:
            批次平均损失（标量）
        """
        batch_size = Y.shape[0]
        # 取真实类别的概率（加1e-10避免log(0)）
        log_probs = np.log(y_pred[range(batch_size), Y] + 1e-10)
        # 平均损失（负的均值）
        return -np.mean(log_probs)

    def backpropagate(self, X, Y, caches, y_pred):
        """
        反向传播（接收输入、标签、缓存，计算各层梯度）
        返回:
            gradients: 各层梯度列表，与self.layers顺序一致，每个元素含(dW, db)
        """
        batch_size = X.shape[0]
        num_layers = len(self.layers)
        gradients = []

        # --------------------------
        # 1. 输出层梯度计算（Softmax + 交叉熵的简化梯度）
        # --------------------------
        last_layer = self.layers[-1]
        last_cache = caches[-1]
        
        # 核心梯度公式：dZ = y_pred - y_true（独热编码等价形式）
        dZ = y_pred.copy()
        dZ[range(batch_size), Y] -= 1
        dZ /= batch_size  # 平均到每个样本（避免批次大小影响梯度幅度）
        
        # 输出层dW和db（A_prev为倒数第二层的A）
        A_prev = last_cache["A_prev"] if num_layers > 1 else X
        dW = np.dot(A_prev.T, dZ)
        db = np.sum(dZ, axis=0, keepdims=True)  # 偏置梯度：按列求和（每个神经元一个偏置）
        
        gradients.append({"dW": dW, "db": db})

        # --------------------------
        # 2. 隐藏层梯度计算（从后往前遍历）
        # --------------------------
        current_dZ = dZ  # 初始为输出层的dZ，反向传递
        for l in range(num_layers - 2, -1, -1):
            layer = self.layers[l]
            cache = caches[l]
            next_layer = self.layers[l + 1]  # 下一层（靠近输出）
            
            # 1. 计算dA：dA = dZ_next · W_next.T
            dA = np.dot(current_dZ, next_layer["W"].T)
            
            # 2. 计算dZ：ReLU导数（Z>0时为1，否则为0）
            dZ = dA * (cache["Z"] > 0)  # 元素-wise乘法
            
            # 3. 计算dW和db（A_prev为当前层的前一层A）
            A_prev = cache["A_prev"] if l > 0 else X
            dW = np.dot(A_prev.T, dZ)
            db = np.sum(dZ, axis=0, keepdims=True)
            
            gradients.append({"dW": dW, "db": db})
            current_dZ = dZ

        # 梯度列表反转：使其与self.layers顺序一致（第0层→第n层）
        gradients = gradients[::-1]
        return gradients

    def update_parameters(self, gradients, learning_rate):
        """用梯度下降更新网络参数"""
        if len(gradients) != len(self.layers):
            raise ValueError("梯度列表长度必须与网络层数一致")
        if learning_rate <= 0:
            raise ValueError("学习率必须为正（推荐0.001~0.01）")
        
        for i in range(len(self.layers)):
            self.layers[i]["W"] -= learning_rate * gradients[i]["dW"]
            self.layers[i]["b"] -= learning_rate * gradients[i]["db"]

    def train(self, epochs, batch_size, learning_rate):
        """
        训练网络（核心改进：批次数据不修改原始数据，仅传递参数）
        参数:
            epochs: 训练轮次（正整数）
            batch_size: 批次大小（正整数，推荐32/64/128）
            learning_rate: 学习率（正小数，推荐0.001~0.01）
        """
        # 参数校验
        if epochs <= 0:
            raise ValueError("训练轮次epochs必须为正整数")
        if batch_size <= 0:
            raise ValueError("批次大小batch_size必须为正整数")
        if batch_size > self.num_train_samples:
            print(f"警告：批次大小({batch_size})超过训练样本数({self.num_train_samples})，将自动使用全量训练")
            batch_size = self.num_train_samples

        # 开始训练
        print(f"\n开始训练：{epochs}轮，批次大小{batch_size}，学习率{learning_rate}")
        for epoch in range(epochs):
            # 1. 随机打乱原始数据（每个epoch重新打乱，保证随机性）
            shuffled_indices = np.random.permutation(self.num_train_samples)
            X_shuffled = self.X_train[shuffled_indices]
            Y_shuffled = self.Y_train[shuffled_indices]
            
            total_loss = 0.0
            batch_count = 0 

            # 2. 分批次训练
            for i in range(0, self.num_train_samples, batch_size):
                # 取当前批次（最后一批可能不足batch_size，自动截断）
                X_batch = X_shuffled[i:i + batch_size]
                Y_batch = Y_shuffled[i:i + batch_size]
                batch_count += 1

                # 3. 前向传播：得到预测概率和中间缓存
                y_pred_batch, caches = self.forward(X_batch)

                # 4. 计算批次损失
                batch_loss = self.cross_entropy_loss(Y_batch, y_pred_batch)
                total_loss += batch_loss

                # 5. 反向传播：计算梯度
                gradients = self.backpropagate(X_batch, Y_batch, caches, y_pred_batch)

                # 6. 更新参数
                self.update_parameters(gradients, learning_rate)

            # 7. 计算并打印本轮平均损失（用实际批次数量，避免误差）
            avg_loss = total_loss / batch_count
            # 每轮训练后打印损失，可选打印训练准确率
            train_acc = self.accuracy(self.X_train[:1000], self.Y_train[:1000])  # 抽样计算准确率，节省时间
            print(f"Epoch {epoch+1}/{epochs} | 平均损失: {avg_loss:.4f} | 训练准确率(抽样): {train_acc:.4f}")

    def predict(self, X):
        """
        预测函数（输入任意数据，返回预测类别）
        参数:
            X: 输入数据，形状(样本数, 特征数)
        返回:
            y_pred_classes: 预测类别，形状(样本数,)
        """
        y_pred_probs, _ = self.forward(X)
        return np.argmax(y_pred_probs, axis=1)  # 取概率最大的类别

    def accuracy(self, X, Y):
        """
        计算准确率（输入数据和真实标签）
        参数:
            X: 输入数据，形状(样本数, 特征数)
            Y: 真实标签，形状(样本数,)
        返回:
            准确率（0~1之间的标量）
        """
        y_pred_classes = self.predict(X)
        return np.mean(y_pred_classes == Y)