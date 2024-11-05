import torch
from cityscapesscripts.helpers.labels import labels
import numpy as np
from scipy.integrate import simps
from scipy.stats import gaussian_kde

def calculate_overlap(kde1, kde2, x, y):
    pdf1 = kde1([x, y])
    pdf2 = kde2([x, y])
    overlap = simps(np.minimum(pdf1, pdf2), axis=0)  # 使用辛普森积分计算重叠区域
    return overlap

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

    return selected_labels, x_data, y_data

def class_overlap(density, x_data_1, y_data_1, x_data_2, y_data_2, selected_labels_1, selected_labels_2, dataset_labels, selected_classes=None):
    unique_labels = np.unique(selected_labels_1)
    unique_labels = unique_labels[unique_labels != 255]  # remove 255 'unlabeled'
    
    if selected_classes is not None:
        # 如果指定了要绘制的特定类别，就只保留这些类别
        unique_labels = [label for label in unique_labels if label in selected_classes]
    
        for i in unique_labels:
            label_info = next(label for label in dataset_labels if label.trainId == i)
            name = label_info.name
            print(name)
            indices = np.where(selected_labels_1 == i)
            x1 = x_data_1[indices]
            y1 = y_data_1[indices]
            
            indices = np.where(selected_labels_2 == i)
            x2 = x_data_2[indices]
            y2 = y_data_2[indices]
    
            reduced_selected_pixel_1 = torch.stack((x1, y1), dim=1)
            reduced_selected_pixel_2 = torch.stack((x2, y2), dim=1)

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
            x = np.linspace(x_min, x_max, density)
            y = np.linspace(y_min, y_max, density)
            
            overlap = calculate_overlap(kde1, kde2, x, y)

            print(f"{name} --> overlap:{overlap*100}%")
    
    
    else:
        x1, y1 = x_data_1, y_data_1
        x2, y2 = x_data_2, y_data_2

    
        reduced_selected_pixel_1 = torch.stack((x1, y1), dim=1)
        reduced_selected_pixel_2 = torch.stack((x2, y2), dim=1)

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
        
        overlap = calculate_overlap(kde1, kde2, x, y)

        print(f" overlap:{overlap}%")


# 使用函数处理不同的数据集和特征
# selected_labels_1, x_1, y_1 = process_data('/home/yliang/work/DAFormer/save_file/encode_decode_feature/baseline/pixel_label/cs_decode_head.concatenated_500_tensor.pt',
#                                             '/home/yliang/work/DAFormer/save_file/encode_decode_feature/baseline/pixel_label/cs_decode_head.concatenated_500_selected_label.npy')

# selected_labels_2, x_2, y_2= process_data('/home/yliang/work/DAFormer/save_file/encode_decode_feature/baseline/pixel_label/gta5_decode_head.concatenated_500_tensor.pt',
#                                             '/home/yliang/work/DAFormer/save_file/encode_decode_feature/baseline/pixel_label/gta5_decode_head.concatenated_500_selected_label.npy')

# selected_labels_3, x_3, y_3= process_data('/home/yliang/work/DAFormer/save_file/encode_decode_feature/HRDA_gta5/pixel_label/cs_decode_head.concatenated_500_tensor.pt',
#                                             '/home/yliang/work/DAFormer/save_file/encode_decode_feature/HRDA_gta5/pixel_label/cs_decode_head.concatenated_500_selected_label.npy')

# selected_labels_4, x_4, y_4 = process_data('/home/yliang/work/DAFormer/save_file/encode_decode_feature/HRDA_gta5/pixel_label/gta5_decode_head.concatenated_500_tensor.pt',
#                                             '/home/yliang/work/DAFormer/save_file/encode_decode_feature/HRDA_gta5/pixel_label/gta5_decode_head.concatenated_500_selected_label.npy')

selected_labels_1, x_1, y_1 = process_data('/home/yliang/work/DAFormer/save_file/encode_decode_feature/baseline/pixel_label/cs_decode_head.concatenated_500_tensor.pt',
                                            '/home/yliang/work/DAFormer/save_file/encode_decode_feature/baseline/pixel_label/cs_decode_head.concatenated_500_selected_label.npy')

selected_labels_2, x_2, y_2= process_data('/home/yliang/work/DAFormer/save_file/encode_decode_feature/baseline/pixel_label/gta5_decode_head.concatenated_500_tensor.pt',
                                            '/home/yliang/work/DAFormer/save_file/encode_decode_feature/baseline/pixel_label/gta5_decode_head.concatenated_500_selected_label.npy')

selected_labels_3, x_3, y_3= process_data('/home/yliang/work/DAFormer/save_file/encode_decode_feature/HRDA_synthia/pixel_label/cs_decode_head.concatenated_500_tensor.pt',
                                            '/home/yliang/work/DAFormer/save_file/encode_decode_feature/HRDA_synthia/pixel_label/cs_decode_head.concatenated_500_selected_label.npy')

selected_labels_4, x_4, y_4 = process_data('/home/yliang/work/DAFormer/save_file/encode_decode_feature/HRDA_synthia/pixel_label/synthia_decode_head.concatenated_500_tensor.pt',
                                            '/home/yliang/work/DAFormer/save_file/encode_decode_feature/HRDA_synthia/pixel_label/synthia_decode_head.concatenated_500_selected_label.npy')






classes=[0,1,2,3,4,5,6,7,8,10,11,12,13,15,17,18]
classes=[10]
# class_1 = class_overlap(20000,x_1, y_1, x_2, y_2, selected_labels_1 ,selected_labels_2, cityscapes_labels, selected_classes = classes)
class_2 = class_overlap(20000,x_3, y_3, x_4, y_4, selected_labels_3 ,selected_labels_4, cityscapes_labels, selected_classes = classes)



#test for same datas
datasets = class_overlap(x_2, y_2, x_2, y_2, selected_labels_2 ,selected_labels_2, cityscapes_labels)




