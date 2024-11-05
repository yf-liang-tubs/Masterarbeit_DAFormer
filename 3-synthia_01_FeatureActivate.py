import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import json
from PIL import Image
from mmseg.models.builder import BACKBONES
import random

import pickle
import os


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

        return image

data_dir = '/home/yliang/work/DAFormer/data/synthia/RAND_CITYSCAPES/RGB'

custom_transform = transforms.Compose([    # Data preprocessing and transformation
    transforms.Resize((256, 512)),
    transforms.ToTensor()
])

image_paths = [os.path.join(data_dir, file) for file in os.listdir(data_dir) if file.endswith('.png')]

# 随机选择 500 张图片的路径
random_image_paths = random.sample(image_paths, 500)
# with open('/home/yliang/work/DAFormer/save_file/output_feature/synthia/500samples_list.pkl', 'wb') as file:
#     pickle.dump(random_image_paths, file)

with open('/home/yliang/work/DAFormer/save_file/output_feature_new/synthia/500samples_list.pkl', 'rb') as file:
    random_image_paths = pickle.load(file)
# 创建一个新的CustomDataset和Dataloader
custom_dataset_random = CustomDataset(random_image_paths, transform=custom_transform)
dataloader= DataLoader(custom_dataset_random, batch_size=1, shuffle=False)


def load_model_from_checkpoint(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    encoder = {}
    selected_backbone = 'mit_b5'
    
    for key, weight in checkpoint['state_dict'].items():
        if 'backbone' in key:
            key = key.replace("backbone.", "") 
            weight = weight.float()
            encoder[key] = weight
    
    model = BACKBONES.get(selected_backbone)(init_cfg = dict(type='Pretrained'))
    model.load_state_dict(encoder)
    
    return model

def extract_features(model, dataloader):
    model.eval()
    outputs = [[] for _ in range(4)]
    
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch
            outputs_batch = model(inputs)
            for i in range(4):
                outputs[i].append(outputs_batch[i])
            print("Batch processed successfully.")
    return [torch.cat(output_list, dim=0) for output_list in outputs]

def save_features(outputs, output_dir, model, dataset):
    for i, output_list in enumerate(outputs):
        print(output_list.size())
        concatenated_tensor = output_list.permute(0, 2, 3, 1)
        torch.save(concatenated_tensor, f'{output_dir}/f{i+1}/f{i+1}_{model}_tensor_{dataset}.pt')
        print(f'saved: {output_dir}/f{i+1}/f{i+1}_{model}_tensor_{dataset}.pt')        


# 2. load model
# Baseline model
baseline_checkpoint_path = '/home/yliang/work/DAFormer/pretrained/BaselineA.pth'
model_baseline = load_model_from_checkpoint(baseline_checkpoint_path)


# Pixmix model
pixmix_checkpoint_path = '/home/yliang/work/DAFormer/pretrained/Pixmix_A.pth'
model_pixmix = load_model_from_checkpoint(pixmix_checkpoint_path)


# starkes GTA5 trained, adapted model
HRDA_checkpoint_path = '/home/yliang/work/DAFormer/pretrained/HRDA_GTA.pth'

HRDA_gta= torch.load(HRDA_checkpoint_path, map_location=torch.device('cpu'))

encoder_HRDA_gta={}
for encoder, weight in HRDA_gta['state_dict'].items():
    if 'model.backbone' in encoder and 'ema_' not in encoder and 'imnet_' not in encoder:
        print(encoder)
        encoder = encoder.replace("model.backbone.","") 
        weight = weight.float()
        encoder_HRDA_gta[encoder] = weight

selected_backbone = 'mit_b5' # import model and set weight
HRDA_gta = BACKBONES.get(selected_backbone)()
HRDA_gta.load_state_dict(encoder_HRDA_gta)


# HRDA_synthia
HRDA_synthia_checkpoint_path = '/home/yliang/work/DAFormer/pretrained/HRDA_Synthia.pth'

HRDA_synthia= torch.load(HRDA_synthia_checkpoint_path, map_location=torch.device('cpu'))

encoder_HRDA_synthia={}
for encoder, weight in HRDA_synthia['state_dict'].items():
    if 'model.backbone' in encoder and 'ema_' not in encoder and 'imnet_' not in encoder:
        print(encoder)
        encoder = encoder.replace("model.backbone.","") 
        weight = weight.float()
        encoder_HRDA_synthia[encoder] = weight

selected_backbone = 'mit_b5' # import model and set weight
HRDA_synthia = BACKBONES.get(selected_backbone)()
HRDA_synthia.load_state_dict(encoder_HRDA_synthia)


# 3.Extract and save features
output_dir = '/home/yliang/work/DAFormer/save_file/output_feature_new/synthia'

outputs_baseline = extract_features(model_baseline, dataloader)
save_features(outputs_baseline, output_dir ,model = 'baseline',dataset='synthia')

outputs_pixmix = extract_features(model_pixmix, dataloader)
save_features(outputs_pixmix, output_dir , model ='pixmix', dataset='synthia')

outputs_HRDA_gta = extract_features(HRDA_gta, dataloader)
save_features(outputs_HRDA_gta, output_dir , model='HRDA_gta', dataset='synthia')

outputs_HRDA_synthia = extract_features(HRDA_synthia, dataloader)
save_features(outputs_HRDA_synthia, output_dir , model='HRDA_synthia', dataset='synthia')




# 4. before tarining model
before_training_model = BACKBONES.get('mit_b5')(init_cfg = dict(type='Pretrained'))
before_training_model.eval()

outputs = [[] for _ in range(4)]
with torch.no_grad():
    for batch in dataloader:
        inputs = batch
        outputs_batch = before_training_model(inputs)
        for i in range(4):
            outputs[i].append(outputs_batch[i])
        print("Batch processed successfully.")
        
outputs_cat = [torch.cat(output_list, dim=0) for output_list in outputs]

for i, output_list in enumerate(outputs_cat):
    print(output_list.size())
    # concatenated_tensor = torch.cat(output_list, dim=0)
    concatenated_tensor = output_list.permute(0, 2, 3, 1)
    torch.save(concatenated_tensor, f'/home/yliang/work/DAFormer/save_file/before_training_feature/SYHTIA/f{i+1}_tensor_SYHTIA.pt')
    print(f'saved: /home/yliang/work/DAFormer/save_file/before_training_feature/SYHTIA/f{i+1}_tensor_SYHTIA.pt')

