from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.combine import SMOTEENN
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np

def kmeans_sampling(X, y, n_clusters=5, random_state=42):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    kmeans.fit(X_scaled)

    cluster_labels = kmeans.labels_
    cluster_centers = kmeans.cluster_centers_

    selected_indices = []

    for i in range(n_clusters):
        cluster_indices = [idx for idx, label in enumerate(cluster_labels) if label == i]
        cluster_center = cluster_centers[i]
        min_distance = float('inf')
        min_distance_idx = -1

        for idx in cluster_indices:
            distance = np.linalg.norm(X_scaled[idx] - cluster_center)
            if distance < min_distance:
                min_distance = distance
                min_distance_idx = idx

        selected_indices.append(min_distance_idx)

    X_selected = X[selected_indices]
    y_selected = y[selected_indices]

    return X_selected, y_selected

def separate_classes(X, y):
    class_data = {}
    for x, label in zip(X, y):
        if label not in class_data:
            class_data[label] = []
        class_data[label].append(x)
    return class_data

def undersampling(X, y, random_state=42):
    rus = RandomUnderSampler(random_state=random_state)
    X_resampled, y_resampled = rus.fit_resample(X, y)
    return X_resampled, y_resampled

def oversampling(X, y, random_state=42):
    ros = RandomOverSampler(random_state=random_state)
    X_resampled, y_resampled = ros.fit_resample(X, y)
    return X_resampled, y_resampled

def combined_sampling(X, y, random_state=42):
    smote_enn = SMOTEENN(random_state=random_state)
    X_resampled, y_resampled = smote_enn.fit_resample(X, y)
    return X_resampled, y_resampled

def resampling_with_kmeans(X, y, strategy='undersampling', majority_ratio=0.9):
    unique_labels = np.unique(y)
    label_counts = [(label, sum(y == label)) for label in unique_labels]
    label_counts.sort(key=lambda x: x[1], reverse=True)

    if strategy == 'undersampling':
        majority_class = label_counts[0][0]
        sampled_classes = label_counts[1:]
    else:
        majority_class = label_counts[-1][0]
        sampled_classes = label_counts[:-1]

    X_majority = [x for x, label in zip(X, y) if label == majority_class]
    X_resampled = list(X_majority)
    y_resampled = [majority_class] * len(X_majority)

    for c, count in sampled_classes:
        X_c = [x for x, label in zip(X, y) if label == c]
        n_clusters = max(2, int(majority_ratio * count))
        kmeans = KMeans(n_clusters=n_clusters, n_init='auto').fit(X_c)
        X_c_selected = kmeans.cluster_centers_

        X_resampled.extend(X_c_selected)
        y_resampled.extend([c] * n_clusters)

    return np.array(X_resampled), np.array(y_resampled)

def undersampling_with_kmeans(X, y, majority_ratio=0.9):
    return resampling_with_kmeans(X, y, strategy='undersampling', majority_ratio=majority_ratio)

def oversampling_with_kmeans(X, y, majority_ratio=0.9):
    return resampling_with_kmeans(X, y, strategy='oversampling', majority_ratio=majority_ratio)

def combined_sampling_with_kmeans(X, y, n_clusters_majority, n_clusters_minority):
    unique_labels, counts = np.unique(y, return_counts=True)
    majority_class = unique_labels[counts.argmax()]
    minority_class = unique_labels[counts.argmin()]
    X_majority = [x for x, label in zip(X, y) if label == majority_class]
    X_minority = [x for x, label in zip(X, y) if label == minority_class]

    kmeans_majority = KMeans(n_clusters=n_clusters_majority).fit(X_majority)
    X_majority_selected = kmeans_majority.cluster_centers_

    kmeans_minority = KMeans(n_clusters=n_clusters_minority).fit(X_minority)
    X_minority_selected = kmeans_minority.cluster_centers_

    X_resampled = list(X_majority_selected)
    y_resampled = [majority_class] * n_clusters_majority

    X_resampled.extend(X_minority_selected)
    y_resampled.extend([minority_class] * n_clusters_minority)

    return np.array(X_resampled), np.array(y_resampled)
