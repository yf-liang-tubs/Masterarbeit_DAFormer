import torch
import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import simps
from scipy.stats import gaussian_kde

def calculate_overlap(kde1, kde2, x, y):
    pdf1 = kde1([x, y])
    pdf2 = kde2([x, y])
    overlap = simps(np.minimum(pdf1, pdf2), axis=0)  # 使用辛普森积分计算重叠区域
    return overlap


def plot_comparison(ax, pixel_file1, labels_file1, pixel_file2, labels_file2, dataset1, dataset2):

    reduced_selected_pixel_1 = torch.load(pixel_file1)
    reduced_selected_pixel_2 = torch.load(pixel_file2)
    
    selected_labels_1 = np.load(labels_file1)
    selected_labels_2 = np.load(labels_file2)

    x_1, y_1 = reduced_selected_pixel_1[:, 0], reduced_selected_pixel_1[:, 1]
    x_2, y_2 = reduced_selected_pixel_2[:, 0], reduced_selected_pixel_2[:, 1]

    unique_labels_1 = np.unique(selected_labels_1)
    unique_labels_1 = unique_labels_1[unique_labels_1 != 255]
    for i in unique_labels_1:
        indices = np.where(selected_labels_1 == i)
        ax.scatter(x_1[indices], y_1[indices], c='red', s=2, alpha=0.5, label=f"{dataset1}" if i == unique_labels_1[0] else "")

    unique_labels_2 = np.unique(selected_labels_2)
    unique_labels_2 = unique_labels_2[unique_labels_2 != 255]
    for i in unique_labels_2:
        indices = np.where(selected_labels_2 == i)
        ax.scatter(x_2[indices], y_2[indices], c='dodgerblue', s=2, alpha=0.5, label=f"{dataset2}" if i == unique_labels_2[0] else "")

    handles, labels = ax.get_legend_handles_labels()
    unique_labels = np.unique(labels)
    ax.legend(handles[:2], unique_labels[:2])
    # ax.set_title(title)

    # 使用高斯核函数进行核密度估计
    reduced_selected_pixel_1_np = reduced_selected_pixel_1.numpy()
    reduced_selected_pixel_2_np = reduced_selected_pixel_2.numpy()
    reduced_selected_pixel_1_np_transpose = reduced_selected_pixel_1_np.T
    reduced_selected_pixel_2_np_transpose = reduced_selected_pixel_2_np.T
    kde1 = gaussian_kde(reduced_selected_pixel_1_np_transpose)
    kde2 = gaussian_kde(reduced_selected_pixel_2_np_transpose)
        
    x_min = min(np.min(reduced_selected_pixel_1_np[:, 0]), np.min(reduced_selected_pixel_2_np[:, 0]))
    x_max = max(np.max(reduced_selected_pixel_1_np[:, 0]), np.max(reduced_selected_pixel_2_np[:, 0]))
    y_min = min(np.min(reduced_selected_pixel_1_np[:, 1]), np.min(reduced_selected_pixel_2_np[:, 1]))
    y_max = max(np.max(reduced_selected_pixel_1_np[:, 1]), np.max(reduced_selected_pixel_2_np[:, 1]))
    x = np.linspace(x_min, x_max, 90000)
    y = np.linspace(y_min, y_max, 90000)
    
    scale_factor = 1
    overlap = calculate_overlap(kde1, kde2, x, y)*scale_factor
    
    # area_smaller_dataset = min(np.ptp(reduced_selected_pixel_1_np[:, 0]) * np.ptp(reduced_selected_pixel_1_np[:, 1]), np.ptp(reduced_selected_pixel_2_np[:, 0]) * np.ptp(reduced_selected_pixel_2_np[:, 1]))
    area_smaller_dataset = min(len(reduced_selected_pixel_1_np), len(reduced_selected_pixel_2_np)) * 2
    max_overlap = (overlap / area_smaller_dataset)
    max_overlap = round(max_overlap, 2)


    print(f"overlap:{overlap}  area_smaller:{area_smaller_dataset}  %:{max_overlap}")

'''
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
plt.savefig(f'/home/yliang/work/DAFormer/save_file/encode_decode_feature/plot/transparent/{model}_{dataset1}_vs_{dataset2}.png', dpi=600, bbox_inches='tight', pad_inches=0.1)
'''

pixel_1 = '/home/yliang/work/DAFormer/save_file/encode_decode_feature/baseline/pixel_label/cs_decode_head.concatenated_500_tensor.pt'
label_1 = '/home/yliang/work/DAFormer/save_file/encode_decode_feature/baseline/pixel_label/cs_decode_head.concatenated_500_selected_label.npy'

pixel_2 = '/home/yliang/work/DAFormer/save_file/encode_decode_feature/baseline/pixel_label/gta5_decode_head.concatenated_500_tensor.pt'
label_2 = '/home/yliang/work/DAFormer/save_file/encode_decode_feature/baseline/pixel_label/gta5_decode_head.concatenated_500_selected_label.npy'


pixel_3 ='/home/yliang/work/DAFormer/save_file/encode_decode_feature/HRDA_gta5/pixel_label/cs_decode_head.concatenated_500_tensor.pt'
label_3 ='/home/yliang/work/DAFormer/save_file/encode_decode_feature/HRDA_gta5/pixel_label/cs_decode_head.concatenated_500_selected_label.npy'

pixel_4 ='/home/yliang/work/DAFormer/save_file/encode_decode_feature/HRDA_gta5/pixel_label/gta5_decode_head.concatenated_500_tensor.pt'
label_4 ='/home/yliang/work/DAFormer/save_file/encode_decode_feature/HRDA_gta5/pixel_label/gta5_decode_head.concatenated_500_selected_label.npy'

pixel_5 ='/home/yliang/work/DAFormer/save_file/encode_decode_feature/HRDA_synthia/pixel_label/cs_decode_head.concatenated_500_tensor.pt'
label_5 ='/home/yliang/work/DAFormer/save_file/encode_decode_feature/HRDA_synthia/pixel_label/cs_decode_head.concatenated_500_selected_label.npy'

pixel_6 ='/home/yliang/work/DAFormer/save_file/encode_decode_feature/HRDA_synthia/pixel_label/synthia_decode_head.concatenated_500_tensor.pt'
label_6 ='/home/yliang/work/DAFormer/save_file/encode_decode_feature/HRDA_synthia/pixel_label/synthia_decode_head.concatenated_500_selected_label.npy'


fig, axs = plt.subplots(1, 2, figsize=(16, 6))

plot_comparison(axs[0], pixel_2, label_2, pixel_2, label_2 ,'CS','GTA5')
plot_comparison(axs[1], pixel_3, label_3, pixel_3, label_3 ,'CS','GTA5')
plot_comparison(axs[1], pixel_5, label_5, pixel_6, label_6 ,'CS','SYNTHIA')

plt.subplots_adjust(left=0.17, wspace=0.1, top=0.8, bottom=0.17)

plt.savefig('/home/yliang/work/DAFormer/save_file/encode_decode_feature/plot/transparent/bl_vs_hrda_gta.pdf', dpi=600, bbox_inches='tight', pad_inches=0.1)
