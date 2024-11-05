import torch
from PIL import Image
import numpy as np
import pickle
import os

data_dir = '/home/yliang/work/DAFormer/data/gta5'

# 4. Skalieren label tensor mit nearest neighbor downscaling auf [H,W,1]
# 创建标签文件路径列表（与随机选择的图像一一对应)
with open('/home/yliang/work/DAFormer/save_file/output_feature_new/gta5/500samples_list.pkl', 'rb') as file:
    random_selected_images = pickle.load(file)

label_dir = '/home/yliang/work/DAFormer/data/gta5/segmentation_trainid/labels'
label_paths = [os.path.join(label_dir, os.path.basename(image_path)) for image_path in random_selected_images]



# Define the downsized size
# desired_size = (16, 8)  # f4
# desired_size = (32, 16) # f3
# desired_size = (64, 32)  # f2
desired_size = (128, 64)  # f1

downscaled_label_tensors = [] # Create an empty list to store the downsized label tensors for each image

print(' opening labels_path')
for image_path in label_paths: 
    
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
torch.save(stacked_label_tensor, '/home/yliang/work/DAFormer/save_file/output_feature/gta5/label_Id_tensor/f4_labels_tensor_gta.pt')
print('labels_tensor saved')

#test
label_paths=['/home/yliang/work/DAFormer/data/gta5/segmentation_trainid/labels/00121.png']
if np.any(label_array == 18):
    print("数字 33 在数组中！")
else:
    print("数字 33 不在数组中！")
    
indices = np.where(label_array == 18)
if len(indices[0]) > 0:
    print("数字 33 在以下位置：")
    for i in range(len(indices[0])):
        print(f"位置 ({indices[0][i]}, {indices[1][i]})")
else:
    print("数字 33 不在数组中！")