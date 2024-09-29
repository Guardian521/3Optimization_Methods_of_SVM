import torch
import torch.nn as nn
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import time

# 乳腺癌数据集
bcancer = load_breast_cancer()
X, y = bcancer.data, bcancer.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=23)

# 标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 转换为Tensor
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)  # 变成列向量
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)  # 变成列向量

# 多项式核函数
def polynomial_kernel(X, Y=None, degree=3, coef0=1):
    if Y is None:
        Y = X
    return (X @ Y.T + coef0) ** degree

# 计算核矩阵
K_train = polynomial_kernel(X_train)
K_test = polynomial_kernel(X_test, X_train)

# 定义SVM模型
class KernelSVM(nn.Module):
    def __init__(self, n_samples):
        super(KernelSVM, self).__init__()
        self.alpha = nn.Parameter(torch.zeros(n_samples, 1))

    def forward(self, K):
        return K @ self.alpha

# 初始化模型
n_samples = X_train.shape[0]
model = KernelSVM(n_samples)

# 超参数
alpha = 0.01
num_epochs = 60
lambda_param = 1.0  # 正则化参数

# 损失函数
def squared_hinge_loss(output, target):
    hinge_loss = torch.maximum(torch.tensor(0.0), 1 - target * output)
    return torch.sum(hinge_loss ** 2)

# SGD
def sgd_step(model, alpha):
    with torch.no_grad():
        for param in model.parameters():
            param -= alpha * param.grad

# 训练模型
sgd_begin = time.time()
for epoch in range(num_epochs):
    model.train()
    # 前向传播
    outputs = model(K_train)
    # 损失
    loss = squared_hinge_loss(outputs, y_train) + lambda_param * torch.sum(model.alpha ** 2)
    # 反向传播
    loss.backward()
    # 裁剪梯度
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    # 更新参数
    sgd_step(model, alpha)
    # 清零梯度
    model.zero_grad()
    print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')
sgd_end = time.time()
print(f"SVM_SGD 的训练时间为 {sgd_end - sgd_begin}")

# 评估模型
model.eval()
with torch.no_grad():
    test_outputs = model(K_test)
    predicted = (test_outputs >= 0).float()
    accuracy = accuracy_score(y_test, predicted)
    print(f'SVM_SGD Accuracy: {accuracy:.4f}')
    print(f'SVM_SGD:{classification_report(y_test, predicted)}')
