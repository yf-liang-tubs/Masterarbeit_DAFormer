import torch
import numpy as np
import matplotlib.pyplot as plt

dataset1, dataset2 ='cityscapes', 'gta5'     # cityscapes/gta5/synthia
model =  'baseline'                         # baseline/pixmix/HRDA_gta
feature = ['f3', 'f4']                         # f1/f2/f3/f4


def plot_comparison(ax, dataset1, dataset2, model, layer, title):

    reduced_selected_pixel_1 = torch.load(f'/home/yliang/work/DAFormer/save_file/output_feature_new/visualize_plot/transparent/{model}/pixels_labels/{model}_{layer}_{dataset1}_tensor.pt')
    reduced_selected_pixel_2 = torch.load(f'/home/yliang/work/DAFormer/save_file/output_feature_new/visualize_plot/transparent/{model}/pixels_labels/{model}_{layer}_{dataset2}_tensor.pt')
    
    selected_labels_1 = np.load(f'/home/yliang/work/DAFormer/save_file/output_feature_new/visualize_plot/transparent/{model}/pixels_labels/{model}_{layer}_{dataset1}_selected_label.npy')
    selected_labels_2 = np.load(f'/home/yliang/work/DAFormer/save_file/output_feature_new/visualize_plot/transparent/{model}/pixels_labels/{model}_{layer}_{dataset2}_selected_label.npy')

    x_1, y_1 = reduced_selected_pixel_1[:, 0], reduced_selected_pixel_1[:, 1]
    x_2, y_2 = reduced_selected_pixel_2[:, 0], reduced_selected_pixel_2[:, 1]

    unique_labels_1 = np.unique(selected_labels_1)
    for i in unique_labels_1:
        indices = np.where(selected_labels_1 == i)
        ax.scatter(x_1[indices], y_1[indices], c='red', s=5, alpha=0.5, label=f"{dataset1}" if i == unique_labels_1[0] else "")

    unique_labels_2 = np.unique(selected_labels_2)
    for i in unique_labels_2:
        indices = np.where(selected_labels_2 == i)
        ax.scatter(x_2[indices], y_2[indices], c='blue', s=5, alpha=0.5, label=f"{dataset2}" if i == unique_labels_2[0] else "")

    handles, labels = ax.get_legend_handles_labels()
    unique_labels = np.unique(labels)
    ax.legend(handles[:2], unique_labels[:2])
    ax.set_title(title)


fig, axs = plt.subplots(2, 2, figsize=(20, 15))
fig.suptitle(f'{model}_{dataset1}_vs_{dataset2}', fontsize=16)


# Use the function to plot comparisons for baseline models
for i, layer in enumerate(feature):
    row = i // 2
    col = i % 2
    sub_title = f'{layer}'
    plot_comparison(axs[row, col], dataset1, dataset2, model, layer, sub_title)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # 调整子图之间的间距和边距

print(f'saved {model}_{dataset1}_vs_{dataset2}.png')
plt.savefig(f'/home/yliang/work/DAFormer/save_file/output_feature_new/visualize_plot/transparent/{model}/{model}_{dataset1}_vs_{dataset2}.png', dpi=300, bbox_inches='tight', pad_inches=0.1)