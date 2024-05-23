import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles

def generate_circular_data(n_samples=10000, noise=0.05, factor=0.5, random_state=None):
    X, y = make_circles(n_samples=n_samples, noise=noise, factor=factor, random_state=random_state)
    return X, y

def save_datasets_to_file(datasets, file_name):
    dataset_dict = {}
    for i, (X, y) in enumerate(datasets):
        dataset_dict[f'X_{i}'] = X
        dataset_dict[f'y_{i}'] = y
    np.savez(file_name, **dataset_dict)

def plot_data_distribution(X, y, title):
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral, edgecolors='k')
    plt.title(title)
    plt.show()

def make_imbalanced(X, y, imbalance_ratio=0.1):
    # 找到第一类和第二类的索引
    class1_indices = np.where(y == 0)[0]
    class2_indices = np.where(y == 1)[0]

    # 随机选择一部分第一类的样本
    np.random.shuffle(class1_indices)
    selected_class1_indices = class1_indices[:int(len(class1_indices)*imbalance_ratio)]

    # 从第二类的样本中获取所有样本
    selected_class2_indices = class2_indices

    # 合并选定的索引
    selected_indices = np.concatenate([selected_class1_indices, selected_class2_indices])

    # 选择对应的样本和标签
    X_imbalanced = X[selected_indices]
    y_imbalanced = y[selected_indices]

    return X_imbalanced, y_imbalanced

# 设置不平衡比例
imbalance_ratios = [0.3, 0.2, 0.1]

datasets = []
for i, imbalance_ratio in enumerate(imbalance_ratios):
    X, y = generate_circular_data(noise=0.1, random_state=i+42)
    X_imbalanced, y_imbalanced = make_imbalanced(X, y, imbalance_ratio=imbalance_ratio)
    datasets.append((X_imbalanced, y_imbalanced))

# 保存数据集到文件
save_datasets_to_file(datasets, 'circular_datasets.npz')

# 绘制样本散点图
for i, (X, y) in enumerate(datasets):
    plot_data_distribution(X, y, f'Dataset {i+1} (imbalance ratio: {imbalance_ratios[i]})')
