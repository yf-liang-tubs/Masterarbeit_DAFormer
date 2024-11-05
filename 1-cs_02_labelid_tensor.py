import torch
from torchvision import transforms
from torchvision.datasets import Cityscapes
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
import os

# 1. Create Data Loader
data_dir = '/home/yliang/work/DAFormer/data/cityscapes' # Path to the dataset folder

custom_transform = transforms.Compose([    # Data preprocessing and transformation
    transforms.Resize((256, 512)),
    transforms.ToTensor()
])

cityscapes_dataset = Cityscapes(data_dir, split='val', mode='fine', target_type='semantic',
                                transform=custom_transform, target_transform=custom_transform)

data_loader = DataLoader(cityscapes_dataset, batch_size=1, shuffle=False)



# 2. Skalieren label tensor mit nearest neighbor downscaling auf [H,W,1]

# Create an empty list to store the downsized label tensors for each image
downscaled_label_tensors = []

# Define the downsized size(Note the order of dimensions: [W, H], Not[H, W])

# desired_size = (16, 8)  # f4
# desired_size = (32, 16) # f3
# desired_size = (64, 32)  # f2
desired_size = (128, 64)  # f1

# Create Labels file name
image_filenames = cityscapes_dataset.images

label_filenames = []
for image_filename in image_filenames:
    base_name = os.path.basename(image_filename)
    city = base_name.split('_')[0]  # 假设城市信息位于文件名的开头部分，并以下划线分隔
    label_filename = os.path.join(data_dir, 'gtFine', 'val', city, base_name.replace('leftImg8bit', 'gtFine_labelTrainIds'))
    label_filenames.append(label_filename)

# Loop through each image
for image_path in label_filenames:  # Assuming image_paths contains paths to 10 images
    
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
stacked_label_tensor = torch.stack(downscaled_label_tensors)# torch.Size([500, H, W])
torch.save(stacked_label_tensor, '/home/yliang/work/DAFormer/save_file/output_feature/cityscapes/label_Id_tensor/f1_labels_tensor_cityscapes.pt')


