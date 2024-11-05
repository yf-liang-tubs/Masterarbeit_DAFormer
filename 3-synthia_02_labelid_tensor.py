import torch
from PIL import Image
import numpy as np
import pickle
import os

# 4. Skalieren label tensor mit nearest neighbor downscaling auf [H,W,1]
# 创建标签文件路径列表（与随机选择的图像一一对应)
with open('/home/yliang/work/DAFormer/save_file/output_feature/synthia/500samples_list.pkl', 'rb') as file:
    random_selected_images = pickle.load(file)


tarinid_image_paths = []

# 遍历原始路径列表，修改路径并添加到新列表中
for path in random_selected_images:
    # 使用字符串的 replace 方法进行路径替换
    new_path = path.replace('RAND_CITYSCAPES/RGB', 'segmentation_trainid/RAND_CITYSCAPES/GT_1_channel/LABELS').replace('.png', '.png')
    tarinid_image_paths.append(new_path)





downscaled_label_tensors = [] # Create an empty list to store the downsized label tensors for each image

# Define the downsized size
# desired_size = (16, 8)  # f4
# desired_size = (32, 16) # f3
# desired_size = (64, 32)  # f2
desired_size = (128, 64)  # f1

print(' opening labels_path')
for image_path in tarinid_image_paths: 
    
    label_image = Image.open(image_path)
    print(' opening --> '+str(image_path))

    # Resize the label image to the desired size using the NEAREST interpolation mode
    downscaled_label_image = label_image.resize(desired_size, Image.NEAREST)

    # Convert the downsized label image to a tensor
    label_array = np.array(downscaled_label_image)
    label_tensor = torch.from_numpy(label_array)

    # Add the downsized label tensor to the list
    downscaled_label_tensors.append(label_tensor)


# Use torch.stack to stack the tensors in the list into a single tensor
stacked_label_tensor = torch.stack(downscaled_label_tensors)# torch.Size([6382, 8, 16])
print('labels_tensor size = '+ str(stacked_label_tensor.size()))
torch.save(stacked_label_tensor, '/home/yliang/work/DAFormer/save_file/output_feature/synthia/label_Id_tensor/f1_labels_tensor_synthia.pt')
print('labels_tensor saved')
