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

def process_data(dataset, model, layer):
    pixels_reduced_2d = torch.load(f'/home/yliang/work/DAFormer/save_file/before_training_feature/visiualize/pixels_labels/{model}/{model}_{layer}_{dataset}_tensor.pt')
    selected_labels = np.load(f'/home/yliang/work/DAFormer/save_file/before_training_feature/visiualize/pixels_labels/{model}/{model}_{layer}_{dataset}_selected_label.npy')
    
    x_data = pixels_reduced_2d[:, 0]
    y_data = pixels_reduced_2d[:, 1]

    point_colors = get_point_colors(selected_labels, cityscapes_labels)

    return selected_labels, x_data, y_data, point_colors

def process_data_not_training(dataset, layer):
    pixels_reduced_2d = torch.load(f'/home/yliang/work/DAFormer/save_file/before_training_feature/visiualize/pixels_labels/{layer}_{dataset}_tensor.pt')
    selected_labels = np.load(f'/home/yliang/work/DAFormer/save_file/before_training_feature/visiualize/pixels_labels/{layer}_{dataset}_selected_label.npy')
    
    x_data = pixels_reduced_2d[:, 0]
    y_data = pixels_reduced_2d[:, 1]

    point_colors = get_point_colors(selected_labels, cityscapes_labels)

    return selected_labels, x_data, y_data, point_colors

def plot_comparison(ax, x_data, y_data, selected_labels, dataset_labels , colors, with_border=True):
    # ax.set_title(title)
    ax.set_xticks([])  # 去掉 x 轴刻度
    ax.set_yticks([])  # 去掉 y 轴刻度
    
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
selected_labels_1, x_1, y_1, point_colors_1 = process_data_not_training(dataset='CS',layer='f4')
selected_labels_2, x_2, y_2, point_colors_2 = process_data_not_training(dataset='GTA5',layer='f4')

selected_labels_3, x_3, y_3, point_colors_3 = process_data(model='baseline', layer='f4', dataset='cityscapes')
selected_labels_4, x_4, y_4, point_colors_4 = process_data(model='baseline', layer='f4', dataset='gta5')



selected_labels_left = np.concatenate((selected_labels_1, selected_labels_2))
x_left=torch.cat((x_1, x_2)); y_left=torch.cat((y_1, y_2))
point_color_left = np.concatenate((point_colors_1, point_colors_2), axis=0)


selected_labels_right = np.concatenate((selected_labels_3, selected_labels_4))
x_right=torch.cat((x_3, x_4)); y_right=torch.cat((y_3, y_4))
point_color_right = np.concatenate((point_colors_3, point_colors_4), axis=0)



# Plot

# create subplot
fig, axs = plt.subplots(1, 2, figsize=(16, 5))
fig.suptitle('T-SNE Feature Space Analysis', fontsize=16)

plot_comparison(axs[0], x_1, y_1, selected_labels_1, cityscapes_labels, point_colors_1, with_border=False)
plot_comparison(axs[0], x_2, y_2, selected_labels_2, cityscapes_labels, point_colors_2,with_border=True)
axs[0].set_title('Before Training')


axs[0].legend(loc='upper right', bbox_to_anchor=(1.0, 1.0), borderaxespad=0.0, ncol=1)
from matplotlib.lines import Line2D
legend_elements_0 = [Line2D([0], [0], marker='o', color='w', label='GTA5', markerfacecolor='blue',markeredgecolor='black', markersize=5, markeredgewidth=0.5),
                   Line2D([0], [0], marker='o', color='w', label='CS', markerfacecolor='blue', markersize=5)]
legend_0 = axs[0].legend(handles=legend_elements_0, loc='lower right', bbox_to_anchor=(1.0, 0.0), borderaxespad=0.0, ncol=1)
axs[0].add_artist(legend_0)



plot_comparison(axs[1], x_3, y_3, selected_labels_3, cityscapes_labels, point_colors_3,with_border=False)
plot_comparison(axs[1], x_4, y_4, selected_labels_4, cityscapes_labels, point_colors_4,with_border=True)
axs[1].set_title('After Training Of Baseline')

axs[1].legend(loc='upper right', bbox_to_anchor=(1.0, 1.0), borderaxespad=0.0, ncol=1)
from matplotlib.lines import Line2D
legend_elements_1 = [Line2D([0], [0], marker='o', color='w', label='GTA5', markerfacecolor='blue',markeredgecolor='black', markersize=5, markeredgewidth=0.5),
                   Line2D([0], [0], marker='o', color='w', label='CS', markerfacecolor='blue', markersize=5)]
legend_1 = axs[1].legend(handles=legend_elements_1, loc='lower right', bbox_to_anchor=(1.0, 0.0), borderaxespad=0.0, ncol=1)
axs[1].add_artist(legend_1)

plt.subplots_adjust(top=0.85)


# last_handles, last_labels = axs[-1, -1].get_legend_handles_labels()
last_handles, last_labels = axs[0].get_legend_handles_labels()


from matplotlib.patches import Patch

last_label_colors = []
for last_label_name in last_labels:
    last_labels_info = next(label for label in cityscapes_labels if label.name == last_label_name)
    color = last_labels_info.color
    last_label_colors.append(color)
last_label_colors_array = np.array(last_label_colors)
last_label_colors_array = last_label_colors_array.reshape(-1, 3)
last_label_colors_array = last_label_colors_array / 255.0

rectangle_handles = [Patch(color=color, label=label) for color, label in zip(last_label_colors_array, last_labels)]

# 移除重复的图例条目
unique_handles, unique_labels = [], []
for handle, label in zip(rectangle_handles, last_labels):
    if label not in unique_labels:
        unique_handles.append(handle)
        unique_labels.append(label)

# Plot legend for the entire figure
fig.legend(unique_handles, unique_labels, loc='lower center', ncol=10, bbox_to_anchor=(0.5, 0.0), borderaxespad=0.02)




# plt.subplots_adjust(left=0.05, wspace=0.3,top=0.95)  # Adjust bottom margin for the legend to fit properly
plt.subplots_adjust(left=0.1, wspace=0.1, top=0.8, bottom=0.2)  

plt.savefig('/home/yliang/work/DAFormer/save_file/before_training_feature/visiualize/plot/test2.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
