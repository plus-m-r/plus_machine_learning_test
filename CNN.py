import numpy as np

def he_initialization(input_dimension, output_dimension):
    """He初始化（适用于ReLU激活的隐藏层）"""
    if input_dimension <= 0 or output_dimension <= 0:
        raise ValueError("输入/输出维度必须为正整数")
    standard_deviation = np.sqrt(2.0 / input_dimension)
    return np.random.randn(input_dimension, output_dimension) * standard_deviation


class ConvolutionalNeuralNetwork:
    def __init__(self, training_data, training_labels):
        """初始化卷积神经网络""" 
        self.training_data = training_data
        self.training_labels = training_labels
        self.network_layers = []
        self.number_of_training_samples = training_data.shape[0]
        self.initial_height, self.initial_width = training_data.shape[2], training_data.shape[3]

    def add_convolution_pooling_layer(self, output_channels, kernel_size, pool_window_size, stride_size=1, padding_size=1):
        """添加卷积+池化层（含参数初始化）"""
        if len(self.network_layers) == 0:
            input_channels = self.training_data.shape[1]
            input_height, input_width = self.initial_height, self.initial_width
        else:
            previous_layer = self.network_layers[-1]
            input_channels = previous_layer["output_channels"]
            input_height, input_width = previous_layer["pooled_height"], previous_layer["pooled_width"]
        
        # 计算卷积输出维度（确保与反向传播一致）
        convolution_output_height = (input_height - kernel_size + 2 * padding_size) // stride_size + 1
        convolution_output_width = (input_width - kernel_size + 2 * padding_size) // stride_size + 1
        
        # 严格校验：确保卷积输出能被池化窗口整除且窗口不截断
        assert convolution_output_height % pool_window_size == 0, \
            f"卷积输出高度{convolution_output_height}不能被池化窗口{pool_window_size}整除"
        assert convolution_output_width % pool_window_size == 0, \
            f"卷积输出宽度{convolution_output_width}不能被池化窗口{pool_window_size}整除"
        assert (input_height + 2*padding_size - kernel_size) % stride_size == 0, \
            f"步长{stride_size}导致卷积高度计算有剩余像素"
        assert (input_width + 2*padding_size - kernel_size) % stride_size == 0, \
            f"步长{stride_size}导致卷积宽度计算有剩余像素"
            
        pooled_height = convolution_output_height // pool_window_size
        pooled_width = convolution_output_width // pool_window_size
        
        # 初始化卷积核权重与偏置
        convolution_weights = he_initialization(
            input_dimension=input_channels * kernel_size * kernel_size,
            output_dimension=output_channels
        ).reshape(output_channels, input_channels, kernel_size, kernel_size)
        convolution_biases = np.zeros((output_channels, 1))
        
        self.network_layers.append({
            "layer_type": "convolution_pooling",
            "weights": convolution_weights,
            "biases": convolution_biases,
            "input_channels": input_channels,
            "output_channels": output_channels,
            "kernel_size": kernel_size,
            "pool_window_size": pool_window_size,
            "stride_size": stride_size,
            "padding_size": padding_size,
            "pooled_height": pooled_height,
            "pooled_width": pooled_width,
            # 缓存卷积输出维度用于反向传播校验
            "conv_output_height": convolution_output_height,
            "conv_output_width": convolution_output_width
        })
        
        print(f"新增卷积+池化层: 输入通道{input_channels} → 输出通道{output_channels} | "
              f"卷积核{kernel_size}×{kernel_size} | 池化窗口{pool_window_size}×{pool_window_size} | "
              f"输出维度{output_channels}×{pooled_height}×{pooled_width}")

    def add_fully_connected_layer(self, output_dimension, is_output_layer=False):
        """添加全连接层（含参数初始化）"""
        if len(self.network_layers) == 0:
            input_dimension = np.prod(self.training_data.shape[1:])
        else:
            previous_layer = self.network_layers[-1]
            if previous_layer["layer_type"] == "convolution_pooling":
                input_dimension = previous_layer["output_channels"] * previous_layer["pooled_height"] * previous_layer["pooled_width"]
            else:
                input_dimension = previous_layer["weights"].shape[1]
        
        assert input_dimension > 0, f"全连接层输入维度必须为正，当前为{input_dimension}"
        
        fully_connected_weights = he_initialization(input_dimension, output_dimension)
        fully_connected_biases = np.zeros((1, output_dimension))
        
        self.network_layers.append({
            "layer_type": "fully_connected",
            "weights": fully_connected_weights,
            "biases": fully_connected_biases,
            "is_output_layer": is_output_layer,
            "input_dimension": input_dimension,
            "output_dimension": output_dimension
        })
        
        layer_role = "输出层" if is_output_layer else "隐藏层"
        print(f"新增全连接层: 输入维度{input_dimension} → 输出维度{output_dimension}（{layer_role}）")

    def pad_input_data(self, input_data, padding_size):
        """对输入图像进行零填充（统一正向/反向传播的填充逻辑）"""
        if padding_size == 0:
            return input_data
        # 明确指定填充位置：仅在高度和宽度方向的前后填充
        return np.pad(
            input_data, 
            pad_width=((0, 0), (0, 0), 
                      (padding_size, padding_size), 
                      (padding_size, padding_size)), 
            mode='constant',
            constant_values=0
        )

    def convolution_forward_propagation(self, input_data, convolution_weights, convolution_biases, stride_size, padding_size):
        """卷积层前向传播"""
        padded_input_data = self.pad_input_data(input_data, padding_size)
        n, c_in, h_in, w_in = padded_input_data.shape
        c_out, _, k_h, k_w = convolution_weights.shape
        
        # 计算输出维度（与反向传播严格一致）
        h_out = (h_in - k_h) // stride_size + 1
        w_out = (w_in - k_w) // stride_size + 1
        
        convolution_output_data = np.zeros((n, c_out, h_out, w_out))
        
        for i in range(n):
            for c in range(c_out):
                for h in range(h_out):
                    for w in range(w_out):
                        # 计算窗口位置（确保与反向传播对称）
                        h_start = h * stride_size
                        h_end = h_start + k_h
                        w_start = w * stride_size
                        w_end = w_start + k_w
                        
                        # 提取输入窗口（确保完整）
                        input_window = padded_input_data[i, :, h_start:h_end, w_start:w_end]
                        assert input_window.shape == (c_in, k_h, k_w), \
                            f"正向传播窗口错误: {input_window.shape} 预期: {(c_in, k_h, k_w)}"
                        
                        convolution_output_data[i, c, h, w] = (
                            np.sum(input_window * convolution_weights[c]) + 
                            convolution_biases[c, 0]
                        )
        
        return convolution_output_data

    def max_pooling_forward_propagation(self, input_data, pool_window_size):
        """最大池化前向传播"""
        n, c, h_in, w_in = input_data.shape
        h_out = h_in // pool_window_size
        w_out = w_in // pool_window_size
        
        pooled_output_data = np.zeros((n, c, h_out, w_out))
        pooling_mask = np.zeros_like(input_data)
        
        for i in range(n):
            for channel in range(c):
                for h in range(h_out):
                    for w in range(w_out):
                        h_start = h * pool_window_size
                        h_end = h_start + pool_window_size
                        w_start = w * pool_window_size
                        w_end = w_start + pool_window_size
                        
                        window = input_data[i, channel, h_start:h_end, w_start:w_end]
                        max_idx = np.unravel_index(np.argmax(window), window.shape)
                        
                        pooling_mask[i, channel, h_start + max_idx[0], w_start + max_idx[1]] = 1
                        pooled_output_data[i, channel, h, w] = np.max(window)
        
        return pooled_output_data, pooling_mask

    def max_pooling_backward_propagation(self, gradient_of_pooled_output, pooling_mask, pool_window_size):
        """最大池化反向传播"""
        gradient_of_previous_input = np.zeros_like(pooling_mask)
        n, c, h_out, w_out = gradient_of_pooled_output.shape
        
        for i in range(n):
            for channel in range(c):
                for h in range(h_out):
                    for w in range(w_out):
                        h_start = h * pool_window_size
                        h_end = h_start + pool_window_size
                        w_start = w * pool_window_size
                        w_end = w_start + pool_window_size
                        
                        gradient_of_previous_input[i, channel, h_start:h_end, w_start:w_end] += \
                            gradient_of_pooled_output[i, channel, h, w] * \
                            pooling_mask[i, channel, h_start:h_end, w_start:w_end]
        
        return gradient_of_previous_input

    def convolution_backward_propagation(self, gradient_of_convolution_output, cache):
        """卷积层反向传播（彻底修复广播错误）"""
        # 1. 提取缓存参数
        previous_input = cache["previous_activation_data"]
        weights = cache["weights"]
        biases = cache["biases"]
        stride = cache["stride_size"]
        padding = cache["padding_size"]
        
        # 2. 获取维度信息
        n, c_in, h_prev, w_prev = previous_input.shape
        c_out, _, k_h, k_w = weights.shape
        _, _, h_out, w_out = gradient_of_convolution_output.shape
        
        # 3. 初始化梯度
        d_prev = np.zeros_like(previous_input)
        d_weights = np.zeros_like(weights)
        d_biases = np.zeros_like(biases)
        
        # 4. 填充输入（确保与正向传播一致）
        padded_prev = self.pad_input_data(previous_input, padding)
        
        # 5. 获取填充后的维度
        padded_h, padded_w = padded_prev.shape[2], padded_prev.shape[3]
        
        # 6. 创建填充后的梯度张量
        padded_d_prev = np.zeros_like(padded_prev)
        
        # 7. 逐元素计算梯度
        for i in range(n):
            for c in range(c_out):
                for h in range(h_out):
                    for w in range(w_out):
                        # 计算窗口位置
                        h_start = h * stride
                        h_end = h_start + k_h
                        w_start = w * stride
                        w_end = w_start + k_w
                        
                        # 获取当前梯度值
                        grad_value = gradient_of_convolution_output[i, c, h, w]
                        
                        # 更新权重梯度
                        input_window = padded_prev[i, :, h_start:h_end, w_start:w_end]
                        d_weights[c] += input_window * grad_value
                        
                        # 更新偏置梯度
                        d_biases[c] += grad_value
                        
                        # 更新前一层输入梯度（在填充后的空间）
                        padded_d_prev[i, :, h_start:h_end, w_start:w_end] += weights[c] * grad_value
        
        # 8. 从填充后的梯度中提取有效区域
        if padding > 0:
            d_prev = padded_d_prev[:, :, padding:-padding, padding:-padding]
        else:
            d_prev = padded_d_prev
        
        d_biases = d_biases.reshape(c_out, 1)
        return d_prev, d_weights, d_biases

    def forward_propagation(self, input_data):
        """完整前向传播"""
        cache = []
        prev_activation = input_data
        
        for layer in self.network_layers:
            if layer["layer_type"] == "convolution_pooling":
                conv_linear = self.convolution_forward_propagation(
                    input_data=prev_activation,
                    convolution_weights=layer["weights"],
                    convolution_biases=layer["biases"],
                    stride_size=layer["stride_size"],
                    padding_size=layer["padding_size"]
                )
                conv_activation = np.maximum(0, conv_linear)  # ReLU激活
                pooled_activation, pool_mask = self.max_pooling_forward_propagation(
                    input_data=conv_activation,
                    pool_window_size=layer["pool_window_size"]
                )
                
                cache.append({
                    "layer_type": "convolution_pooling",
                    "previous_activation_data": prev_activation,
                    "convolution_activation_data": conv_activation,
                    "pooled_activation_data": pooled_activation,
                    "pooling_mask": pool_mask,
                    "weights": layer["weights"],
                    "biases": layer["biases"],
                    "stride_size": layer["stride_size"],
                    "padding_size": layer["padding_size"],
                    "pool_window_size": layer["pool_window_size"]
                })
                
                prev_activation = pooled_activation

            elif layer["layer_type"] == "fully_connected":
                if len(prev_activation.shape) > 2:
                    flattened = prev_activation.reshape(prev_activation.shape[0], -1)
                else:
                    flattened = prev_activation
                
                fc_linear = np.dot(flattened, layer["weights"]) + layer["biases"]
                
                if layer["is_output_layer"]:
                    # Softmax激活
                    max_val = np.max(fc_linear, axis=1, keepdims=True)
                    exp_val = np.exp(fc_linear - max_val)
                    sum_exp = np.sum(exp_val, axis=1, keepdims=True)
                    sum_exp = np.maximum(sum_exp, 1e-10)
                    fc_activation = exp_val / sum_exp
                else:
                    fc_activation = np.maximum(0, fc_linear)  # ReLU激活
                
                cache.append({
                    "layer_type": "fully_connected",
                    "flattened_previous_activation": flattened,
                    "fully_connected_activation_data": fc_activation,
                    "weights": layer["weights"],
                    "biases": layer["biases"]
                })
                
                prev_activation = fc_activation

        return prev_activation, cache

    def cross_entropy_loss_calculation(self, true_labels, predicted_probabilities):
        """计算交叉熵损失"""
        n = true_labels.shape[0]
        log_probs = np.log(predicted_probabilities[range(n), true_labels] + 1e-10)
        return -np.mean(log_probs)

    def backward_propagation(self, input_data, true_labels, cache, predictions):
        """完整反向传播"""
        n = input_data.shape[0]
        num_layers = len(self.network_layers)
        grads = []
        
        # 输出层梯度
        d_linear = predictions.copy()
        d_linear[range(n), true_labels] -= 1
        d_linear /= n
        
        if self.network_layers[-1]["layer_type"] == "fully_connected":
            last_cache = cache[-1]
            d_weights = np.dot(last_cache["flattened_previous_activation"].T, d_linear)
            d_biases = np.sum(d_linear, axis=0, keepdims=True)
            grads.append({"gradient_of_weights": d_weights, "gradient_of_biases": d_biases})
            current_grad = np.dot(d_linear, self.network_layers[-1]["weights"].T)
        else:
            raise ValueError("输出层必须是全连接层")

        # 反向传播其他层
        for i in range(num_layers - 2, -1, -1):
            layer = self.network_layers[i]
            layer_cache = cache[i]
            
            if layer["layer_type"] == "fully_connected":
                d_linear = current_grad * (layer_cache["fully_connected_activation_data"] > 0)
                d_weights = np.dot(layer_cache["flattened_previous_activation"].T, d_linear)
                d_biases = np.sum(d_linear, axis=0, keepdims=True)
                grads.append({"gradient_of_weights": d_weights, "gradient_of_biases": d_biases})
                current_grad = np.dot(d_linear, layer["weights"].T)

            elif layer["layer_type"] == "convolution_pooling":
                # 重塑梯度以匹配池化层输出维度
                h, w = layer["pooled_height"], layer["pooled_width"]
                c = layer["output_channels"]
                d_pool = current_grad.reshape(n, c, h, w)
                
                # 池化层反向传播
                d_conv = self.max_pooling_backward_propagation(
                    gradient_of_pooled_output=d_pool,
                    pooling_mask=layer_cache["pooling_mask"],
                    pool_window_size=layer["pool_window_size"]
                )
                
                # 卷积层反向传播（应用ReLU导数）
                d_conv_activation = d_conv * (layer_cache["convolution_activation_data"] > 0)
                d_prev, d_weights, d_biases = self.convolution_backward_propagation(
                    gradient_of_convolution_output=d_conv_activation,
                    cache=layer_cache
                )
                
                grads.append({"gradient_of_weights": d_weights, "gradient_of_biases": d_biases})
                current_grad = d_prev

        return grads[::-1]

    def update_network_parameters(self, parameter_gradients, learning_rate):
        """梯度下降参数更新"""
        for i in range(len(self.network_layers)):
            self.network_layers[i]["weights"] -= learning_rate * parameter_gradients[i]["gradient_of_weights"]
            self.network_layers[i]["biases"] -= learning_rate * parameter_gradients[i]["gradient_of_biases"]

    def train_network(self, training_epochs, number_of_samples_per_batch, learning_rate):
        """完整训练流程"""
        if training_epochs <= 0:
            raise ValueError("训练轮次必须为正整数")
        if number_of_samples_per_batch <= 0:
            raise ValueError("批次大小必须为正整数")
        
        n = self.number_of_training_samples
        if number_of_samples_per_batch > n:
            number_of_samples_per_batch = n
            print(f"警告：批次大小调整为{number_of_samples_per_batch}（等于训练样本数）")

        print(f"\n开始训练：{training_epochs}轮，批次大小{number_of_samples_per_batch}，学习率{learning_rate}")
        for epoch in range(training_epochs):
            indices = np.random.permutation(n)
            shuffled_data = self.training_data[indices]
            shuffled_labels = self.training_labels[indices]
            
            total_loss = 0.0
            batches = 0

            for start in range(0, n, number_of_samples_per_batch):
                end = start + number_of_samples_per_batch
                batch_data = shuffled_data[start:end]
                batch_labels = shuffled_labels[start:end]
                batches += 1

                preds, cache = self.forward_propagation(batch_data)
                loss = self.cross_entropy_loss_calculation(batch_labels, preds)
                total_loss += loss
                
                grads = self.backward_propagation(batch_data, batch_labels, cache, preds)
                self.update_network_parameters(grads, learning_rate)

            avg_loss = total_loss / batches
            sample_indices = np.random.choice(n, min(1000, n), replace=False)
            acc = self.calculate_accuracy(
                self.training_data[sample_indices], 
                self.training_labels[sample_indices]
            )
            
            print(f"第{epoch+1}/{training_epochs}轮 | 平均损失: {avg_loss:.4f} | 训练准确率(抽样): {acc:.4f}")

    def predict_classes(self, input_data):
        """预测函数"""
        preds, _ = self.forward_propagation(input_data)
        return np.argmax(preds, axis=1)

    def calculate_accuracy(self, input_data, true_labels):
        """计算准确率"""
        return np.mean(self.predict_classes(input_data) == true_labels)
    