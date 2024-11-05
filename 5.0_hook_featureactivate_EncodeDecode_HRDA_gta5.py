import torch
import numpy as np
import random
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import Cityscapes
from mmseg.models.builder import BACKBONES
from mmseg.models.builder import SEGMENTORS

import pickle
from PIL import Image


if torch.cuda.is_available():
    device = torch.device('cuda')
    print('CUDA is available. Using GPU...')
else:
    device = torch.device('cpu')
    print('CUDA is not available. Using CPU...')


# Dataloader---------------------------------------

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(0)
np.random.seed(0)
random.seed(0)

batch_size = 1
'''
# cs
# Path to the dataset folder
data_dir = '/home/yliang/work/DAFormer/data/cityscapes'

# Data preprocessing and transformation
custom_transform = transforms.Compose([
    transforms.Resize((256, 512)),
    transforms.ToTensor()
])

# Create the Cityscapes dataset
cityscapes_dataset = Cityscapes(data_dir, split='val', mode='fine', target_type='semantic',
                                transform=custom_transform, target_transform=custom_transform)


cs_dataloader = DataLoader(cityscapes_dataset,
                           pin_memory=True,
                        batch_size=batch_size, 
                        shuffle=False,
                        worker_init_fn=seed_worker,
                        generator=g)
'''


# gta5
# 1. Create Data Loader
class CustomDataset(Dataset):
    def __init__(self, image_paths, transform=None, num_samples=None):
        self.image_paths = image_paths
        self.transform = transform
        self.num_samples = num_samples if num_samples is not None else len(self.image_paths)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]

        image = Image.open(image_path)

        if self.transform is not None:
            image = self.transform(image)
            
        img_meta = {
            'width': image.shape[2],  # 修改此行，使用 image.shape 获取图像宽度
            'height': image.shape[1],  # 修改此行，使用 image.shape 获取图像高度
            # 添加其他需要的图像元信息
        }

        return image, img_meta
    
data_dir = '/home/yliang/work/DAFormer/data/gta5'

custom_transform = transforms.Compose([    # Data preprocessing and transformation
    transforms.Resize((256, 512)),
    transforms.ToTensor()
])

with open('/home/yliang/work/DAFormer/save_file/output_feature_new/gta5/500samples_list.pkl', 'rb') as file:
    random_selected_images = pickle.load(file)

# 创建一个新的CustomDataset和Dataloader
custom_dataset_random = CustomDataset(random_selected_images, transform=custom_transform)
gta5_dataloader= DataLoader(custom_dataset_random,pin_memory=True,
                        batch_size=batch_size, 
                        shuffle=False,
                        worker_init_fn=seed_worker,
                        generator=g)

# Load model
# 2.HRDA-----------------------------------------------------------------
hrda_decode_encode = SEGMENTORS.get('EncoderDecoder')(
    backbone = dict(type='mit_b5',init_cfg=dict(type='Pretrained')),
    decode_head= dict(type='DAFormerHead' ,init_cfg=dict(type='Pretrained'),
                          in_channels=[64, 128, 320, 512],  channels=256,  
                          num_classes=19, in_index=[0, 1, 2, 3],
                        decoder_params=dict(
                        embed_dims=256, embed_cfg=dict(type='mlp', act_cfg=None, norm_cfg=None),
                        embed_neck_cfg=dict(type='mlp', act_cfg=None, norm_cfg=None),
                        fusion_cfg=dict(type='aspp',sep=True, dilations=(1, 6, 12, 18),
                                        pool=False, act_cfg=dict(type='ReLU'),
                                        norm_cfg=dict(type='BN', requires_grad=True)
                                )),
                            norm_cfg=dict(
                                    type='BN',
                                    requires_grad=True
                                )
                        ))


HRDA_state_dict = torch.load('/home/yliang/work/DAFormer/model_info/HRDA_state_dict.pt')

hrda_decode_encode.load_state_dict(HRDA_state_dict)
hrda_gta5_decode_encode=hrda_decode_encode.to(device)
hrda_gta5_decode_encode.eval()

#Test---------------------------------------------------
# input_data = torch.randn(1, 3, 256, 512).to(device)

# forward dummy
# forward_dummy_result = hrda_gta5_decode_encode.forward_dummy(input_data)
# torch.save(forward_dummy_result, f'/home/yliang/work/DAFormer/save_file/encode_decode_feature/HRDA/forward_dummy_result.pt')
forward_dummy_result = torch.load('/home/yliang/work/DAFormer/save_file/encode_decode_feature/HRDA/forward_dummy_result.pt', map_location=device)




def activate_and_extract_features(model, dataloader, target_layer_names):
    activations = {layer_name: [] for layer_name in target_layer_names}
    hooks = []

    def get_activation(name):
        def hook(model, input, output):
            activations[name].append(output.detach())
        return hook

    for layer_name in target_layer_names:
        target_layer = dict(model.named_modules())[layer_name]
        # target_layer = getattr(model, layer_name)
        hook = target_layer.register_forward_hook(get_activation(layer_name))
        hooks.append(hook)

    batch_counter = 0
    with torch.no_grad():
        for data in dataloader:
            if batch_counter >= 500:
                break
            input_data, img_metas= data
            inputs = input_data.to(device)
            # img_metas.to(device) # cs
            img_metas = {key: value.to(device) for key, value in img_metas.items()} #gta5
      
            model.encode_decode(inputs,img_metas)
            
            batch_counter += 1
            if batch_counter % 10 == 0:
                print(f"{model} : Processed {batch_counter} batches")
            
            del inputs

    for hook in hooks:
        hook.remove()

    return activations



layer_names = []
for name, module in hrda_gta5_decode_encode.named_modules():
    print(name)
    if 'fuse_layer.bottleneck.bn' in name:
        layer_names.append(name)
# print(layer_names)

dataset_dataloader = gta5_dataloader  #cs_dataloader/gta5_dataloader

extracted_features_HDRA = activate_and_extract_features(hrda_gta5_decode_encode, dataset_dataloader, layer_names)
torch.save(extracted_features_HDRA, f'/home/yliang/work/DAFormer/save_file/encode_decode_feature/HRDA/gta5_decode_head.fuse_layer.bottleneck.bn_batch500.pt')
print('Saved: extracted_features')

