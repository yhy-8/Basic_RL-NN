import numpy as np
import gzip
import matplotlib.pyplot as plt
import os


# ==========================================
# 1. 数据加载与预处理部分
# ==========================================

def load_mnist_images(filename):
    """加载MNIST图像文件"""
    with gzip.open(filename, 'rb') as f:
        # MNIST文件头描述：前16字节是魔数、图片数量、行数、列数
        # 从第16字节开始读取数据
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    # 将数据重塑为 (图片数量, 28*28)
    # 归一化：将像素值从 0-255 缩放到 0-1 之间，这对神经网络训练至关重要
    return data.reshape(-1, 784).astype(np.float32) / 255.0


def load_mnist_labels(filename):
    """加载MNIST标签文件"""
    with gzip.open(filename, 'rb') as f:
        # 前8字节是魔数和标签数量
        # 从第8字节开始读取数据
        data = np.frombuffer(f.read(), np.uint8, offset=8)
    return data


def one_hot_encode(y, num_classes=10):
    """将标签转换为独热编码 (One-Hot Encoding)"""
    # 例如：标签 3 变成 [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    return np.eye(num_classes)[y]


# 设置文件路径（请确保文件在当前目录下，或者修改为你的绝对路径）
files = {
    'train_img': 'mnist/train-images-idx3-ubyte.gz',
    'train_lbl': 'mnist/train-labels-idx1-ubyte.gz',
    'test_img': 'mnist/t10k-images-idx3-ubyte.gz',
    'test_lbl': 'mnist/t10k-labels-idx1-ubyte.gz'
}

print("正在加载数据...")
# 检查文件是否存在
for k, v in files.items():
    if not os.path.exists(v):
        print(f"错误：找不到文件 {v}，请确保文件在当前目录下。")
        exit()

X_train = load_mnist_images(files['train_img'])
y_train_raw = load_mnist_labels(files['train_lbl'])
X_test = load_mnist_images(files['test_img'])
y_test_raw = load_mnist_labels(files['test_lbl'])

# 转换标签为 One-Hot 格式
y_train = one_hot_encode(y_train_raw)
y_test = one_hot_encode(y_test_raw)

print(f"训练集形状: {X_train.shape}, 标签形状: {y_train.shape}")
print(f"测试集形状: {X_test.shape}, 标签形状: {y_test.shape}")


# ==========================================
# 2. 开头展示训练集图片
# ==========================================

def show_examples(images, labels):
    plt.figure(figsize=(10, 5))
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        # 将 784 维向量还原回 28x28 用于显示
        img = images[i].reshape(28, 28)
        plt.imshow(img, cmap='gray')
        plt.title(f"Label: {labels[i]}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()


print("展示前10张训练图片...")
show_examples(X_train, y_train_raw)


# ==========================================
# 3. 神经网络类定义 (BP核心)
# ==========================================

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1):
        self.lr = learning_rate

        # 初始化权重和偏置
        # 使用正态分布随机初始化，乘以0.01是为了让数值较小，利于收敛
        # W1: 输入层 -> 隐藏层1 (784 -> 16)
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))

        # W2: 隐藏层1 -> 隐藏层2 (16 -> 16)
        self.W2 = np.random.randn(hidden_size, hidden_size) * 0.01
        self.b2 = np.zeros((1, hidden_size))

        # W3: 隐藏层2 -> 输出层 (16 -> 10)
        self.W3 = np.random.randn(hidden_size, output_size) * 0.01
        self.b3 = np.zeros((1, output_size))

    def sigmoid(self, z):
        """Sigmoid 激活函数"""
        return 1 / (1 + np.exp(-z))

    def sigmoid_derivative(self, a):
        """Sigmoid 的导数，输入 a 是已经经过 sigmoid 激活的值"""
        return a * (1 - a)

    def softmax(self, z):
        """Softmax 激活函数 (用于多分类输出)"""
        # 减去最大值是为了防止 exp 计算溢出
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def forward(self, X):
        """前向传播"""
        # 第一层 (Input -> Hidden 1)
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = self.sigmoid(self.Z1)

        # 第二层 (Hidden 1 -> Hidden 2)
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = self.sigmoid(self.Z2)

        # 第三层 (Hidden 2 -> Output)
        self.Z3 = np.dot(self.A2, self.W3) + self.b3
        self.A3 = self.softmax(self.Z3)

        return self.A3

    def backward(self, X, y):
        """反向传播 (BP算法核心)"""
        m = X.shape[0]  # 样本数量 (batch size)

        # 1. 计算输出层误差 (Cross Entropy Loss + Softmax 的导数简化为 A3 - y)
        dZ3 = self.A3 - y

        # 计算 W3 和 b3 的梯度
        dW3 = np.dot(self.A2.T, dZ3) / m
        db3 = np.sum(dZ3, axis=0, keepdims=True) / m

        # 2. 计算隐藏层2的误差
        # 链式法则: dZ2 = (dZ3 * W3.T) * sigmoid_derivative(A2)
        dA2 = np.dot(dZ3, self.W3.T)
        dZ2 = dA2 * self.sigmoid_derivative(self.A2)

        # 计算 W2 和 b2 的梯度
        dW2 = np.dot(self.A1.T, dZ2) / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m

        # 3. 计算隐藏层1的误差
        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * self.sigmoid_derivative(self.A1)

        # 计算 W1 和 b1 的梯度
        dW1 = np.dot(X.T, dZ1) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m

        # 4. 更新参数 (梯度下降)
        self.W3 -= self.lr * dW3
        self.b3 -= self.lr * db3
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

    def compute_loss(self, y_pred, y_true):
        """计算全局交叉熵损失"""
        m = y_true.shape[0]
        # 添加一个微小值 1e-9 防止 log(0)
        log_probs = -np.log(y_pred[range(m), np.argmax(y_true, axis=1)] + 1e-9)
        loss = np.sum(log_probs) / m
        return loss


# ==========================================
# 4. 训练与测试循环
# ==========================================

# 超参数设置
input_size = 784  # 28*28
hidden_size = 16  # 题目要求每层16个
output_size = 10  # 0-9 十个数字
learning_rate = 0.5  # 学习率
epochs = 100  # 训练轮数
batch_size = 64  # 批量大小 (Mini-batch SGD)

# 初始化网络
nn = NeuralNetwork(input_size, hidden_size, output_size, learning_rate)

print(f"\n开始训练... (Epochs: {epochs}, Batch Size: {batch_size}, LR: {learning_rate})")

# 训练循环
for epoch in range(epochs):
    # 打乱数据
    permutation = np.random.permutation(X_train.shape[0])
    X_train_shuffled = X_train[permutation]
    y_train_shuffled = y_train[permutation]

    for i in range(0, X_train.shape[0], batch_size):
        # 获取当前 batch
        X_batch = X_train_shuffled[i:i + batch_size]
        y_batch = y_train_shuffled[i:i + batch_size]

        # 前向传播
        output = nn.forward(X_batch)

        # 反向传播并更新权重
        nn.backward(X_batch, y_batch)

    # 每个 Epoch 结束后评估一次
    train_output = nn.forward(X_train)
    loss = nn.compute_loss(train_output, y_train)

    # 计算准确率
    predictions = np.argmax(train_output, axis=1)
    labels = np.argmax(y_train, axis=1)
    accuracy = np.mean(predictions == labels)

    print(f"Epoch {epoch + 1}/{epochs} - Loss: {loss:.4f} - Accuracy: {accuracy:.4f}")

# ==========================================
# 5. 最终测试集验证
# ==========================================

print("\n正在测试集上评估...")
test_output = nn.forward(X_test)
test_predictions = np.argmax(test_output, axis=1)
test_labels = np.argmax(y_test, axis=1)
test_acc = np.mean(test_predictions == test_labels)

print(f"测试集最终准确率: {test_acc * 100:.2f}%")

# 随机挑几个测试集图片看看预测结果
print("\n随机抽取测试集预测展示:")
indices = np.random.choice(len(X_test), 5, replace=False)
plt.figure(figsize=(10, 3))
for i, idx in enumerate(indices):
    plt.subplot(1, 5, i + 1)
    img = X_test[idx].reshape(28, 28)
    plt.imshow(img, cmap='gray')
    pred = test_predictions[idx]
    true_label = test_labels[idx]
    plt.title(f"Pred: {pred}\nTrue: {true_label}", color=("green" if pred == true_label else "red"))
    plt.axis('off')
plt.show()