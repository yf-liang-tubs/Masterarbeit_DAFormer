import torch
import numpy as np
i
from mmseg.models.builder import BACKBONES



def print_and_save_weights_dimensions(checkpoint_path, save_path):
    # 加载 checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # 打开文件以保存维度信息
    with open(save_path, 'w') as file:
        # 遍历每个参数
        for key, value in checkpoint['state_dict'].items():
            # 打印参数名称和形状
            # print(f"Parameter Name: {key} \nShape: {value.shape} \n{'='*50}")

            # 将信息写入文件
            file.write(f"Parameter Name: {key} \nShape: {value.shape} \n{'='*50}\n")


# 执行函数
print_and_save_weights_dimensions(checkpoint_path='/home/yliang/work/DAFormer/pretrained/HRDA_GTA.pth', 
                                  save_path='/home/yliang/work/DAFormer/model_info/HRDA_GTA.txt')

print_and_save_weights_dimensions(checkpoint_path='/home/yliang/work/DAFormer/pretrained/HRDA_Synthia.pth', 
                                  save_path='/home/yliang/work/DAFormer/model_info/HRDA_Synthia.txt')
