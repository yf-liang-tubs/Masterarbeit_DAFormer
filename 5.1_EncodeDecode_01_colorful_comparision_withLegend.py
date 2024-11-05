import torch
from cityscapesscripts.helpers.labels import labels
import numpy as np
import matplotlib.pyplot as plt

def get_point_colors(selected_labels, dataset_labels:list):
    labels_colors = []
    for train_id in selected_labels:
        label_info = next(label for label in dataset_labels if label.trainId == train_id)
        color = label_info.color
        labels_colors.append(color)

    colors_array = np.array(labels_colors)
    colors_array = colors_array.reshape(-1, 3)
    return colors_array / 255.0

cityscapes_labels = labels


def process_data(pixel_file1, labels_file1):
    pixels_reduced_2d = torch.load(pixel_file1)
    selected_labels = np.load(labels_file1)
    
    x_data = pixels_reduced_2d[:, 0]
    y_data = pixels_reduced_2d[:, 1]

    point_colors = get_point_colors(selected_labels, cityscapes_labels)

    return selected_labels, x_data, y_data, point_colors


def plot_comparison(ax, x_data, y_data, selected_labels, dataset_labels , colors, with_border=True):
    # ax.set_title(title)
    # ax.set_xticks([])  # 去掉 x 轴刻度
    # ax.set_yticks([])  # 去掉 y 轴刻度
    
    unique_labels = np.unique(selected_labels)
    unique_labels = unique_labels[unique_labels != 255]  # remove 255 'unlabeled'
    for i in unique_labels:
        label_info = next(label for label in dataset_labels if label.trainId == i)
        name = label_info.name
        indices = np.where(selected_labels == i)
        ax.scatter(x_data[indices], y_data[indices], c=[colors[indices[0][0]]], s=2, label=f'{name}')
        
        if with_border:
            ax.scatter(x_data[indices], y_data[indices], c=[colors[indices[0][0]]], s=2, label=f'{name}', marker='o', edgecolors='black', linewidths=0.1)
        else:
            ax.scatter(x_data[indices], y_data[indices], c=[colors[indices[0][0]]], s=2, label=f'{name}', marker='o')
    # ax.legend(loc='upper right')



# 使用函数处理不同的数据集和特征
selected_labels_1, x_1, y_1, point_colors_1 = process_data('/home/yliang/work/DAFormer/save_file/encode_decode_feature/HRDA_synthia/pixel_label/cs_decode_head.concatenated_500_tensor.pt',
                                                           '/home/yliang/work/DAFormer/save_file/encode_decode_feature/HRDA_synthia/pixel_label/cs_decode_head.concatenated_500_selected_label.npy')

selected_labels_2, x_2, y_2, point_colors_2 = process_data('/home/yliang/work/DAFormer/save_file/encode_decode_feature/HRDA_synthia/pixel_label/synthia_decode_head.concatenated_500_tensor.pt',
                                                           '/home/yliang/work/DAFormer/save_file/encode_decode_feature/HRDA_synthia/pixel_label/synthia_decode_head.concatenated_500_selected_label.npy')


# Plot

# create subplot
fig, axs = plt.subplots(1, 2, figsize=(16, 6))
# fig.suptitle('T-SNE Feature Space Analysis', fontsize=16)

plot_comparison(axs[0], x_1, y_1, selected_labels_1, cityscapes_labels, point_colors_1, with_border=False)
axs[0].set_title('Before Domain-Generalization')



plot_comparison(axs[1], x_2, y_2, selected_labels_2, cityscapes_labels, point_colors_2,with_border=False)
axs[1].set_title('After Domain-Generalization')

last_handles, last_labels = axs[0].get_legend_handles_labels()


last_label_colors = []
for last_label_name in last_labels:
    last_labels_info = next(label for label in cityscapes_labels if label.name == last_label_name)
    color = last_labels_info.color
    last_label_colors.append(color)
last_label_colors_array = np.array(last_label_colors)
last_label_colors_array = last_label_colors_array.reshape(-1, 3)
last_label_colors_array = last_label_colors_array / 255.0

from matplotlib.patches import FancyBboxPatch
rectangle_handles = [FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0.1", linewidth=0.5, edgecolor='black', facecolor=color) for color in last_label_colors_array]

# 移除重复的图例条目
unique_handles, unique_labels = [], []
for handle, label in zip(rectangle_handles, last_labels):
    if label not in unique_labels:
        unique_handles.append(handle)
        unique_labels.append(label)



# Plot legend for the entire figure
# fig.legend(unique_handles, unique_labels, loc='lower center', ncol=10, bbox_to_anchor=(0.5, 0.0), borderaxespad=0.02)
fig.legend(unique_handles, unique_labels, loc='lower center',
           ncol=10, bbox_to_anchor=(0.5, 0.0), borderaxespad=0.02,
           frameon=False,
           labelspacing = 0)


plt.subplots_adjust(left=0.17, wspace=0.1, top=0.8, bottom=0.17)  

plt.savefig('/home/yliang/work/DAFormer/save_file/encode_decode_feature/plot/generalization/HRDA_gta.pdf', dpi=600, bbox_inches='tight', pad_inches=0.1)

