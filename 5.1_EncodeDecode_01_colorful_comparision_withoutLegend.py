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

def plot_comparison(ax, x_data, y_data, selected_labels, dataset_labels , colors,  selected_classes=None, with_border=True):
    # ax.set_title(title)
    # ax.set_xticks([])  # 去掉 x 轴刻度
    # ax.set_yticks([])  # 去掉 y 轴刻度
    
    unique_labels = np.unique(selected_labels)
    unique_labels = unique_labels[unique_labels != 255]  # remove 255 'unlabeled'
    
    if selected_classes is not None:
        unique_labels = [label for label in unique_labels if label in selected_classes]
    
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
selected_labels_1, x_1, y_1, point_colors_1 = process_data('/home/yliang/work/DAFormer/save_file/encode_decode_feature/baseline/pixel_label/cs_decode_head.concatenated_500_tensor.pt',
                                                           '/home/yliang/work/DAFormer/save_file/encode_decode_feature/baseline/pixel_label/cs_decode_head.concatenated_500_selected_label.npy')

selected_labels_2, x_2, y_2, point_colors_2 = process_data('/home/yliang/work/DAFormer/save_file/encode_decode_feature/baseline/pixel_label/gta5_decode_head.concatenated_500_tensor.pt',
                                                           '/home/yliang/work/DAFormer/save_file/encode_decode_feature/baseline/pixel_label/gta5_decode_head.concatenated_500_selected_label.npy')

selected_labels_3, x_3, y_3, point_colors_3 = process_data('/home/yliang/work/DAFormer/save_file/encode_decode_feature/HRDA_gta5/pixel_label/cs_decode_head.concatenated_500_tensor.pt',
                                                           '/home/yliang/work/DAFormer/save_file/encode_decode_feature/HRDA_gta5/pixel_label/cs_decode_head.concatenated_500_selected_label.npy')

selected_labels_4, x_4, y_4, point_colors_4 = process_data('/home/yliang/work/DAFormer/save_file/encode_decode_feature/HRDA_gta5/pixel_label/gta5_decode_head.concatenated_500_tensor.pt',
                                                           '/home/yliang/work/DAFormer/save_file/encode_decode_feature/HRDA_gta5/pixel_label/gta5_decode_head.concatenated_500_selected_label.npy')



classes = [1]
# Plot
# create subplot
fig, axs = plt.subplots(1, 2, figsize=(16, 6))
# fig.suptitle('DIDEX w/ HRDA', fontsize=16, y=0.9)

plot_comparison(axs[0], x_1, y_1, selected_labels_1, cityscapes_labels, point_colors_1, classes, with_border=False)
plot_comparison(axs[0], x_2, y_2, selected_labels_2, cityscapes_labels, point_colors_2, classes, with_border=True)
# axs[0].set_title('Before Training')

axs[0].legend(loc='upper right', bbox_to_anchor=(1.0, 1.0), borderaxespad=0.0, ncol=1)
from matplotlib.lines import Line2D
legend_elements_0 = [Line2D([0], [0], marker='o', color='w', label='GTA5', markerfacecolor='lightskyblue',markeredgecolor='black', markersize=5, markeredgewidth=0.5),
                Line2D([0], [0], marker='o', color='w', label='CS', markerfacecolor='lightskyblue', markersize=5)]
legend_0 = axs[0].legend(handles=legend_elements_0, loc='lower right', bbox_to_anchor=(1.0, 0.0), borderaxespad=0.0, ncol=1)
axs[0].add_artist(legend_0)


plot_comparison(axs[1], x_3, y_3, selected_labels_3, cityscapes_labels, point_colors_3, classes, with_border=False)
plot_comparison(axs[1], x_4, y_4, selected_labels_4, cityscapes_labels, point_colors_4, classes, with_border=True)
# axs[1].set_title('After Training')

axs[1].legend(loc='upper right', bbox_to_anchor=(1.0, 1.0), borderaxespad=0.0, ncol=1)
from matplotlib.lines import Line2D
legend_elements_1 = [Line2D([0], [0], marker='o', color='w', label='GTA5', markerfacecolor='lightskyblue',markeredgecolor='black', markersize=5, markeredgewidth=0.5),
                Line2D([0], [0], marker='o', color='w', label='CS', markerfacecolor='lightskyblue', markersize=5)]
legend_1 = axs[1].legend(handles=legend_elements_1, loc='lower right', bbox_to_anchor=(1.0, 0.0), borderaxespad=0.0, ncol=1)
axs[1].add_artist(legend_1)

plt.subplots_adjust(left=0.17, wspace=0.1, top=0.8, bottom=0.17)  

plt.savefig(f'/home/yliang/work/DAFormer/save_file/encode_decode_feature/plot/generalization/DIDEX_w_HRDA_cs_gta5_{classes}.png', dpi=600, bbox_inches='tight', pad_inches=0.1)
print(f'saved {classes}')


unique_labels_i = np.unique(selected_labels_1)
unique_labels_i = unique_labels_i[unique_labels_i != 255]  # remove 255 'unlabeled'