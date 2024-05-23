import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.preprocessing import MinMaxScaler
from matplotlib.ticker import FixedLocator
from sklearn.decomposition import PCA

def generate_imbalanced_data(n_classes=3, class_sep=2, weights=None, n_informative=3, n_redundant=1, n_features=20,
                             n_samples=10000, random_state=42):
    X, y = make_classification(n_classes=n_classes, class_sep=class_sep, weights=weights, n_informative=n_informative,
                               n_redundant=n_redundant, flip_y=0, n_features=n_features, n_clusters_per_class=1,
                               n_samples=n_samples, random_state=random_state)
    return X, y

def save_datasets_to_file(datasets, file_name):
    dataset_dict = {}
    for i, (X, y) in enumerate(datasets):
        dataset_dict[f'X_{i}'] = X
        dataset_dict[f'y_{i}'] = y
    np.savez(file_name, **dataset_dict)

def plot_features_radar_charts(datasets):
    for i, (X, y) in enumerate(datasets):
        scaler = MinMaxScaler()
        X_normalized = scaler.fit_transform(X)
        mean_features = []
        labels = [f'Feature {j}' for j in range(X_normalized.shape[1])]
        for class_id in np.unique(y):
            class_data = X_normalized[y == class_id]
            mean_features.append(np.mean(class_data, axis=0))

        num_classes = len(np.unique(y))
        fig, axs = plt.subplots(1, num_classes, figsize=(5 * num_classes, 4), subplot_kw=dict(polar=True))
        for j, features in enumerate(mean_features):
            draw_radar_chart(axs[j], features.tolist(), labels, f'Dataset {i} - Class {j}')
        plt.tight_layout()
        plt.show()

def draw_radar_chart(ax, data, labels, title):
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    data += data[:1]
    angles += angles[:1]

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, color='grey', size=8)
    ax.set_rlabel_position(30)
    ax.yaxis.grid(True, color="grey", linestyle='-')
    ax.xaxis.grid(True, color="grey", linestyle='-')
    ax.yaxis.set_major_locator(FixedLocator(np.linspace(0, 1, 6))) # 设置y轴刻度位置
    ax.set_yticklabels(range(1, 7), color="grey", size=7)
    ax.set_ylim(0, 1)

    ax.plot(angles, data, linewidth=1, linestyle='solid')
    ax.fill(angles, data, 'b', alpha=0.25)
    ax.set_title(title, size=11, color='blue', y=1.1)

def plot_data_distribution(X, y, title):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap=plt.cm.Spectral, edgecolors='k')
    plt.title(title)
    plt.show()

# 生成三个具有不同权重分布的不平衡数据集
weights_list = [
    [0.1, 0.3, 0.6],
    [0.05, 0.35, 0.6],
    [0.2, 0.2, 0.6]
]

datasets = [generate_imbalanced_data(weights=weights, random_state=i+42) for i, weights in enumerate(weights_list)]

# 保存数据集到文件
save_datasets_to_file(datasets, 'imbalanced_datasets.npz')

# 绘制特征雷达图
plot_features_radar_charts(datasets)

# 绘制样本散点图（PCA降维后）
for i, weights in enumerate(weights_list):
    X, y = generate_imbalanced_data(n_classes=3, class_sep=2, weights=weights, n_informative=3, n_redundant=1, n_features=20, n_samples=10000, random_state=42)
    datasets.append((X, y))
    plot_data_distribution(X, y, f'Dataset {i+1} (weights: {weights})')
