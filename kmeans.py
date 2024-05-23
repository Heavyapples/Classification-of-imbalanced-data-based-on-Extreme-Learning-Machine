import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from matplotlib.font_manager import FontProperties
# 使用黑体字体
font = FontProperties(fname=r"C:\Windows\Fonts\simhei.ttf", size=14)
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置全局字体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号'-'显示为方块的问题

# 读取数据
data = pd.read_csv('mammography.csv')
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# 标准化数据
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 使用KMeans进行聚类
kmeans = KMeans(n_clusters=10)
kmeans.fit(X_scaled)

# 获取聚类标签和中心点
labels = kmeans.labels_
centers = kmeans.cluster_centers_

# 原始样本的散点图
plt.figure(figsize=(12, 8))
plt.subplot(221)
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c='black')
plt.title('原始样本的散点图')

# 决策图
plt.subplot(222)
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c='black')
plt.scatter(centers[:, 0], centers[:, 1], color='red')
plt.title('决策图')

# 筛选子类中心点图
plt.subplot(223)
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels)
plt.scatter(centers[:, 0], centers[:, 1], color='red', s=200, marker='X')
plt.title('筛选子类中心点图')

# 划分后子类图
plt.subplot(224)
for i in np.unique(labels):
    plt.scatter(X_scaled[labels == i, 0], X_scaled[labels == i, 1], label=f'Cluster {i+1}')
plt.legend()
plt.title('划分后子类图')

plt.tight_layout()
plt.show()

# 最终聚类效果图
plt.subplot(235)
for i in np.unique(y):
    plt.scatter(X_scaled[y == i, 0], X_scaled[y == i, 1], label=f'Class {i}')
plt.legend()
plt.title('最终聚类效果图')

plt.tight_layout()
plt.show()
