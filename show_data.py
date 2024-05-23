import numpy as np
import pandas as pd

def display_data_from_npz(file_name, n_datasets=3, n_rows=10):
    data = np.load(file_name)

    for i in range(n_datasets):
        X = data[f'X_{i}']
        y = data[f'y_{i}']

        # 将特征和标签合并为一个DataFrame
        df = pd.DataFrame(X, columns=[f'feature_{j}' for j in range(X.shape[1])])
        df['label'] = y

        # 显示数据集的前n行
        print(f'Dataset {i+1} from npz file:')
        print(df.head(n_rows))
        print('\n')

# 从npz文件中加载并显示数据
display_data_from_npz('imbalanced_datasets.npz')
