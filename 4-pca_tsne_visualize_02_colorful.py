import torch
from cityscapesscripts.helpers.labels import labels
import numpy as np
import matplotlib.pyplot as plt
'''
def get_pixels_labels(model, layer, dataset):

    return selected_labels, pixels_reduced_2d  # or whatever result you need
'''
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
    pixels_reduced_2d = torch.load(f'/home/yliang/work/DAFormer/save_file/output_feature_new/visualize_plot/transparent/{model}/pixels_labels/{model}_{layer}_{dataset}_tensor.pt')
    selected_labels = np.load(f'/home/yliang/work/DAFormer/save_file/output_feature_new/visualize_plot/transparent/{model}/pixels_labels/{model}_{layer}_{dataset}_selected_label.npy')
    
    x_data = pixels_reduced_2d[:, 0]
    y_data = pixels_reduced_2d[:, 1]

    point_colors = get_point_colors(selected_labels, cityscapes_labels)

    return selected_labels, x_data, y_data, point_colors

def plot_comparison(ax, x_data, y_data, selected_labels, dataset_labels , colors, title):
    # ax.set_title(title)
    # ax.set_xticks([])  # 去掉 x 轴刻度
    # ax.set_yticks([])  # 去掉 y 轴刻度
    
    unique_labels = np.unique(selected_labels)
    unique_labels = unique_labels[unique_labels != 255]  # remove 255 'unlabeled'
    for i in unique_labels:
        label_info = next(label for label in dataset_labels if label.trainId == i)
        name = label_info.name
        indices = np.where(selected_labels == i)
        ax.scatter(x_data[indices], y_data[indices], c=[colors[indices[0][0]]], s=5, label=f'{name}')
        
    # ax.legend(loc='upper right')


dataset1 ='cityscapes'
dataset2 = 'gta5'
# dataset3 = 'synthia'

model = 'HRDA_gta'  # baseline/pixmix/HRDA_gta

# choose which feature you are interesed 'f1' , 'f2', 'f3' , 'f4'
feature1 = 'f1'
feature2 = 'f2'
feature3 = 'f3'
feature4 = 'f4'

# 使用函数处理不同的数据集和特征
selected_labels_1, x_1, y_1, point_colors_1 = process_data(f'{dataset1}', f'{model}',f'{feature1}')
selected_labels_2, x_2, y_2, point_colors_2 = process_data(f'{dataset2}', f'{model}', f'{feature1}')

selected_labels_3, x_3, y_3, point_colors_3= process_data(f'{dataset1}', f'{model}',f'{feature2}')
selected_labels_4, x_4, y_4, point_colors_4= process_data(f'{dataset2}',f'{model}',f'{feature2}')

selected_labels_5, x_5, y_5, point_colors_5= process_data(f'{dataset1}',f'{model}',f'{feature3}')
selected_labels_6, x_6, y_6, point_colors_6= process_data(f'{dataset2}',f'{model}',f'{feature3}')

selected_labels_7, x_7, y_7, point_colors_7= process_data(f'{dataset1}',f'{model}',f'{feature4}')
selected_labels_8, x_8, y_8, point_colors_8= process_data(f'{dataset2}',f'{model}',f'{feature4}')



# Plot

# create subplot
fig, axs = plt.subplots(4, 2, figsize=(16, 20))
fig.suptitle('T-SNE Feature Space Analysis Cityscapes And GTA5 Of HRDA', fontsize=16)

plot_comparison(axs[0, 0], x_1, y_1, selected_labels_1, cityscapes_labels, point_colors_1, 'f1 of CS')
plot_comparison(axs[0, 1], x_2, y_2, selected_labels_2, cityscapes_labels, point_colors_2, 'f1 of GTA5')


plot_comparison(axs[1, 0], x_3, y_3, selected_labels_3, cityscapes_labels, point_colors_3, 'f2 of CS')
plot_comparison(axs[1, 1], x_4, y_4, selected_labels_4, cityscapes_labels, point_colors_4, 'f2 of GTA5')


plot_comparison(axs[2, 0], x_5, y_5, selected_labels_5, cityscapes_labels, point_colors_5, 'f3 of CS')
plot_comparison(axs[2, 1], x_6, y_6, selected_labels_6, cityscapes_labels, point_colors_6, 'f3 of GTA5')

plot_comparison(axs[3, 0], x_7, y_7, selected_labels_7, cityscapes_labels, point_colors_7, 'f4 of CS')
plot_comparison(axs[3, 1], x_8, y_8, selected_labels_8, cityscapes_labels, point_colors_8, 'f4 of GTA5')

# last_handles, last_labels = axs[-1, -1].get_legend_handles_labels()
last_handles, last_labels = axs[0, 0].get_legend_handles_labels()

'''
# Remove duplicate legend entries
unique_handles, unique_labels = [], []
for handle, label in zip(last_handles, last_labels):
    if label not in unique_labels:
        unique_handles.append(handle)
        unique_labels.append(label)

# Plot legend for the entire figure
# fig.legend(unique_handles, unique_labels, loc='center right', ncol=1, bbox_to_anchor=(1, 0.5))
fig.legend(unique_handles, unique_labels, loc='lower right', ncol=1, bbox_to_anchor=(1, 0.5), fancybox=True, shadow=True)
'''
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
# fig.legend(unique_handles, unique_labels, loc='lower center', ncol=10, bbox_to_anchor=(0.5, 0.0), borderaxespad=0.02)




# plt.subplots_adjust(left=0.05, wspace=0.3,top=0.95)  # Adjust bottom margin for the legend to fit properly
plt.subplots_adjust(left=0.1, wspace=0.1, top=0.95, bottom=0.2)  

plt.savefig(f'/home/yliang/work/DAFormer/save_file/output_feature_new/visualize_plot_final/colorful/{dataset1}_and_{dataset2}_for_{model}.pdf', dpi=600, bbox_inches='tight', pad_inches=0.1)
