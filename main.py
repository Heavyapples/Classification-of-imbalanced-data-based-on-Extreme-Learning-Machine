import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sampling_methods import undersampling, oversampling, combined_sampling, oversampling_with_kmeans, undersampling_with_kmeans, combined_sampling_with_kmeans
from elm import ELM
from sklearn.metrics import confusion_matrix

def plot_data_distribution(X, y, title):
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral, edgecolors='k')
    plt.title(title)
    plt.show()

# 加载数据集
data = np.load('circular_datasets.npz')

# 获取数据集
X_0 = data['X_0']
y_0 = data['y_0']
X_1 = data['X_1']
y_1 = data['y_1']
X_2 = data['X_2']
y_2 = data['y_2']

# 对数据集进行采样处理
X_0_under, y_0_under = undersampling(X_0, y_0)
X_1_over, y_1_over = oversampling(X_1, y_1)
X_2_comb, y_2_comb = combined_sampling(X_2, y_2)

# 创建ELM模型实例
elm = ELM()

# 使用处理后的数据集训练和评估ELM模型
datasets = [(X_0_under, y_0_under, "Undersampled"),
            (X_1_over, y_1_over, "Oversampled"),
            (X_2_comb, y_2_comb, "Combined Sampling")]

for i, (X, y, dataset_name) in enumerate(datasets):
    print(f"Dataset {i}: {dataset_name}")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # 绘制处理后的数据集散点图
    plot_data_distribution(X_train, y_train, f"{dataset_name} Training Data")
    elm.fit(X_train, y_train)
    y_pred = elm.predict(X_test)
    accuracy = elm.score(X_test, y_test)
    print(f"Accuracy: {accuracy:.2f}")
    # 计算混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    # 计算每个类别的准确率
    for j in range(cm.shape[0]):
        class_accuracy = cm[j, j] / cm[j, :].sum()
        print(f"Class {j} Accuracy: {class_accuracy:.2f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("\n")

# 对数据集进行与聚类任务相结合的采样处理
X_0_under, y_0_under = undersampling_with_kmeans(X_0, y_0)
X_0_over, y_0_over = oversampling_with_kmeans(X_0, y_0)
X_0_comb, y_0_comb = combined_sampling_with_kmeans(X_0, y_0, n_clusters_majority=10, n_clusters_minority=5)

# 创建ELM模型实例
elm = ELM()

# 使用处理后的数据集训练和评估ELM模型
datasets = [(X_0_under, y_0_under, "Undersampling with kmeans"),
            (X_1_over, y_1_over, "Oversampling with kmeans"),
            (X_2_comb, y_2_comb, "Combined Sampling with kmeans")]

for i, (X, y, dataset_name) in enumerate(datasets):
    print(f"Dataset {i}: {dataset_name}")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # 绘制处理后的数据集散点图
    plot_data_distribution(X_train, y_train, f"{dataset_name} Training Data")
    elm.fit(X_train, y_train)
    y_pred = elm.predict(X_test)
    accuracy = elm.score(X_test, y_test)
    print(f"Accuracy: {accuracy:.2f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("\n")
