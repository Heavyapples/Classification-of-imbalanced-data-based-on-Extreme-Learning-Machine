import numpy as np
from scipy.special import expit
from sklearn.metrics import accuracy_score
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelBinarizer

# 定义 ELM 类，继承自 BaseEstimator 和 ClassifierMixin
class ELM(BaseEstimator, ClassifierMixin):
    # 构造函数，初始化 ELM 的参数
    def __init__(self, n_hidden_units=100, activation='sigmoid', random_state=None):
        self.n_hidden_units = n_hidden_units  # 隐藏层神经元个数
        self.activation = activation  # 激活函数
        self.random_state = random_state  # 随机数生成器的种子
        self._label_binarizer = LabelBinarizer()  # 用于将标签转换为二进制表示

    # 初始化权重矩阵 W 和偏置矩阵 b
    def _initialize_weights(self, n_features):
        rng = np.random.default_rng(self.random_state)  # 创建随机数生成器
        self.W = rng.uniform(-1, 1, (n_features, self.n_hidden_units))  # 初始化权重矩阵
        self.b = rng.uniform(-1, 1, self.n_hidden_units)  # 初始化偏置矩阵

    # 计算隐藏层的输出
    def _hidden_layer(self, X):
        G = np.dot(X, self.W) + self.b  # 计算隐藏层的加权和
        # 根据激活函数类型进行激活
        if self.activation == 'sigmoid':
            return expit(G)  # 使用 Sigmoid 激活函数
        else:
            raise ValueError(f"Unknown activation function: {self.activation}")

    # 训练模型
    def fit(self, X, y):
        self._initialize_weights(X.shape[1])  # 初始化权重和偏置
        H = self._hidden_layer(X)  # 计算隐藏层的输出
        H_pinv = np.linalg.pinv(H)  # 计算 H 的伪逆矩阵
        self._label_binarizer.fit(y)  # 训练 LabelBinarizer
        T = self._label_binarizer.transform(y)  # 将标签转换为二进制表示
        self.beta = np.dot(H_pinv, T)  # 计算输出权重 beta

    # 预测
    def predict(self, X):
        H = self._hidden_layer(X)  # 计算隐藏层的输出
        T_pred = np.dot(H, self.beta)  # 计算输出层的加权和
        y_pred = self._label_binarizer.inverse_transform(T_pred)  # 将二进制表示转换回原始标签
        return y_pred

    # 评估模型
    def score(self, X, y):
        y_pred = self.predict(X)  # 预测标签
        return accuracy_score(y, y_pred)  # 计算预测准确率
