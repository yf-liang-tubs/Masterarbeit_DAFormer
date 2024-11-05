import torch
from cityscapesscripts.helpers.labels import labels
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

'''
cs_label_tensor = torch.load(label_tensor_file)
flat_labels = cs_label_tensor.flatten()

max_class = 1000
selected_indices = []

np.random.seed(42)
for label in np.unique(flat_labels):
    indices = np.where(flat_labels == label)[0]
    if len(indices) > max_class:
        selected_indices.extend(np.random.choice(indices, max_class, replace=False))
    else:
        selected_indices.extend(indices)


# 从 pixel_data 中选择所选的像素点
bl_output = torch.load(activata_feature_tensor_file)
bl_output_reshaped = bl_output.reshape(-1, bl_output.size()[-1])
selected_pixels = bl_output_reshaped[selected_indices]

# 使用 PCA 进行降维
pca = PCA(n_components=30)  # 假设降维到30维
pixels_reduced = pca.fit_transform(selected_pixels)

# 使用 t-SNE 进行进一步降维到2维
tsne = TSNE(n_components=2)
pixels_reduced_2d = tsne.fit_transform(pixels_reduced)
'''

def dim_reduce(label_tensor_file:str, activata_feature_tensor_file:str):
    """
    Description of the function.

    Parameters:
    label_tensor_file (str): the path of label tensor.
    activata_feature_tensor_file (str): the path of activation feature.
    ...

    Returns:
    selected_indices(list): indices which corresponds to selected pixels.
    pixels_reduced_2d(ndarray): 2 dimensional vectors of selected pixels.
    """
    cs_label_tensor = torch.load(label_tensor_file)
    flat_labels = cs_label_tensor.flatten()

    max_class = 1000
    selected_indices = []

    np.random.seed(42)
    for label in np.unique(flat_labels):
        indices = np.where(flat_labels == label)[0]
        if len(indices) > max_class:
            selected_indices.extend(np.random.choice(indices, max_class, replace=False))
        else:
            selected_indices.extend(indices)


    # 从 pixel_data 中选择所选的像素点
    bl_output = torch.load(activata_feature_tensor_file)
    bl_output_reshaped = bl_output.reshape(-1, bl_output.size()[-1])
    selected_pixels = bl_output_reshaped[selected_indices]

    # 使用 PCA 进行降维
    pca = PCA(n_components=30)  # 假设降维到30维
    pixels_reduced = pca.fit_transform(selected_pixels)
    print('pca reduced successed')

    # 使用 t-SNE 进行进一步降维到2维
    print('tSNE reducing')
    tsne = TSNE(n_components=2)
    pixels_reduced_2d = tsne.fit_transform(pixels_reduced)

    selected_labels = flat_labels[selected_indices]

    # Return statement
    return selected_labels, pixels_reduced_2d  # or whatever result you need


label_tensor_file = '/home/yliang/work/DAFormer/save_file/output_feature/cityscapes/label_Id_tensor/f1_labels_tensor_cityscapes.pt'
activata_feature_tensor_file = '/home/yliang/work/DAFormer/save_file/output_feature/cityscapes/f1/f1_HRDA_gta_tensor_cityscapes.pt'

selected_labels, pixels_reduced_2d = dim_reduce(label_tensor_file, activata_feature_tensor_file)

# 提取x和y坐标
x = pixels_reduced_2d[:, 0]
y = pixels_reduced_2d[:, 1]

cityscapes_labels = labels


labels_colors = []
for train_id in selected_labels:
    label_info = next(label for label in cityscapes_labels if label.trainId == train_id)
    color = label_info.color
    labels_colors.append(color)

colors_array = np.array(labels_colors)
colors_array = colors_array.reshape(-1, 3)
point_colors = colors_array / 255.0

# 创建一个散点图
plt.figure(figsize=(16, 12))

unique_labels = np.unique(selected_labels)
for i in unique_labels:
    label_info = next(label for label in cityscapes_labels if label.trainId == i)
    name = label_info.name
    indices = np.where(selected_labels == i)
    plt.scatter(x[indices], y[indices], c=[point_colors[indices[0][0]]], s=5, label = f'{name}')

   
plt.legend(loc="best")
plt.title('Visualization of model baseline in cityscapes')
plt.savefig('/home/yliang/work/DAFormer/save_file/output_feature/cityscapes/f1/cs_bl_f1_MAX1000_test.png', dpi=150, bbox_inches='tight', pad_inches=0.1)
