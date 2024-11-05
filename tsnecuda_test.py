import torch
import numpy as np
from tsnecuda import TSNE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# 创建一些示例数据
X = np.random.rand(100, 10)

# 使用 tsnecuda 进行 t-SNE 计算
tsne = TSNE(n_components=2)
X_tsne = tsne.fit_transform(X)

print(X_tsne)
