import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gzip
import os
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader

# ================= 1. GPU 设置 =================
# PyTorch 需要手动检查并指定设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"正在使用的设备: {device}")
if device.type == 'cuda':
    print(f"显卡型号: {torch.cuda.get_device_name(0)}")


# ================= 2. 加载本地数据的函数 (逻辑不变) =================
def load_local_mnist(path='.'):
    files = [
        'mnist/train-labels-idx1-ubyte.gz', 'mnist/train-images-idx3-ubyte.gz',
        'mnist/t10k-labels-idx1-ubyte.gz', 'mnist/t10k-images-idx3-ubyte.gz'
    ]
    paths = [os.path.join(path, f) for f in files]

    # 检查文件
    for p in paths:
        if not os.path.exists(p):
            raise FileNotFoundError(f"找不到文件: {p}")

    print("正在加载本地数据...")
    with gzip.open(paths[0], 'rb') as lbpath:
        y_train = np.frombuffer(lbpath.read(), np.uint8, offset=8)
    with gzip.open(paths[1], 'rb') as imgpath:
        x_train = np.frombuffer(imgpath.read(), np.uint8, offset=16).reshape(len(y_train), 28, 28)
    with gzip.open(paths[2], 'rb') as lbpath:
        y_test = np.frombuffer(lbpath.read(), np.uint8, offset=8)
    with gzip.open(paths[3], 'rb') as imgpath:
        x_test = np.frombuffer(imgpath.read(), np.uint8, offset=16).reshape(len(y_test), 28, 28)

    return x_train, y_train, x_test, y_test


x_train_np, y_train_np, x_test_np, y_test_np = load_local_mnist()

# ================= 2.1展示 10 张训练图片 =================
print("展示 10 张训练图片...")
plt.figure(figsize=(12, 5))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_train_np[i], cmap='gray')
    plt.title(f"Label: {y_train_np[i]}")
    plt.axis('off')
plt.suptitle("First 10 Training Images")
plt.show()

# ================= 3. 数据预处理与转为 Tensor =================
# 归一化并转为 Tensor (PyTorch 默认 float32)
# 注意：PyTorch 的全连接层通常需要展平，或者在网络里展平。
# 这里我们保持 (N, 28, 28)
x_train = torch.tensor(x_train_np / 255.0, dtype=torch.float32)
y_train = torch.tensor(y_train_np, dtype=torch.long)  # 标签要是 long 类型
x_test = torch.tensor(x_test_np / 255.0, dtype=torch.float32)
y_test = torch.tensor(y_test_np, dtype=torch.long)

# 制作 DataLoader(关键)
train_ds = TensorDataset(x_train, y_train)
test_ds = TensorDataset(x_test, y_test)

train_loader = DataLoader(train_ds, batch_size=640, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=1000)


# ================= 4. 搭建神经网络模型 =================
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, 10)  # 输出层不需要 Softmax，因为 Loss 函数里自带了

    def forward(self, x):
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        logits = self.fc3(x)
        return logits


# 实例化模型并搬运到 GPU !!!
model = SimpleNet().to(device)
print("\n模型结构:")
print(model)

# ================= 5. 定义损失函数和优化器 =================
criterion = nn.CrossEntropyLoss()  # 自带 Softmax + Log + NLLLoss
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ================= 6. 训练模型 =================
epochs = 150
print("\n开始训练 (PyTorch)...")

for epoch in range(epochs):
    model.train()  # 切换到训练模式
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        # !!! 关键步骤：把数据搬运到 GPU !!!
        images, labels = images.to(device), labels.to(device)

        # 1. 清空梯度
        optimizer.zero_grad()
        # 2. 前向传播
        outputs = model(images)
        # 3. 计算损失
        loss = criterion(outputs, labels)
        # 4. 反向传播(修改隐性参数.grad)
        loss.backward()
        # 5. 更新参数(使用隐性参数.grad)
        optimizer.step()

        running_loss += loss.item()

        # 计算训练集准确率 (可选)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(
        f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(train_loader):.4f}, Accuracy: {100 * correct / total:.2f}%")

# ================= 7. 测试集评估 =================
print("\n在测试集上评估...")
model.eval()  # 切换到评估模式
correct = 0
total = 0

# 不需要计算梯度，节省显存
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"测试集准确率: {100 * correct / total:.2f}%")

# ================= 8.训练结束后展示 10 张预测照片 =================
print("\n展示 10 张预测结果...")
model.eval()

# 取测试集前 10 张
sample_imgs = x_test[:10]
sample_labels = y_test[:10]

# 预测
with torch.no_grad():
    # 搬运到 GPU
    sample_imgs_device = sample_imgs.to(device)
    # 前向传播
    outputs = model(sample_imgs_device)
    # 获取预测结果
    _, predicted = torch.max(outputs, 1)

# 转回 CPU 以便绘图
sample_imgs_np = sample_imgs.numpy()
predicted_np = predicted.cpu().numpy()
sample_labels_np = sample_labels.numpy()

plt.figure(figsize=(12, 5))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(sample_imgs_np[i], cmap='gray')

    # 设置颜色：预测正确为绿色，错误为红色
    color = 'green' if predicted_np[i] == sample_labels_np[i] else 'red'

    plt.title(f"Pred: {predicted_np[i]}\nTrue: {sample_labels_np[i]}", color=color)
    plt.axis('off')

plt.suptitle("Model Predictions on Test Images")
plt.show()
