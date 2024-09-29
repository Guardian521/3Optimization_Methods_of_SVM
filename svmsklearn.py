import torch
import torch.nn as nn
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
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


# 定义SVM模型
class SVM(nn.Module):
    def __init__(self, n_features):
        super(SVM, self).__init__()
        self.linear = nn.Linear(n_features, 1, bias=False)

    def forward(self, x):
        return self.linear(x)
# 初始化模型
n_features = X_train.shape[1]
model = SVM(n_features)




#sklearn实现
svc_begin=time.time()
svm_model = SVC(kernel='linear', C=1.0, random_state=22,max_iter=-1, verbose=True)
svm_model.fit(X_train, y_train)
svc_end=time.time()
print(f"SVC 的训练时间为{svc_end-svc_begin}")

y_pred3 = svm_model.predict(X_test)
accuracy_svc=accuracy_score(y_test, y_pred3)

print(f"SVC Accuracy:, {accuracy_svc:.4f}")
print("Classification Report:")
print(f'SVC:{classification_report(y_test, y_pred3)}')
