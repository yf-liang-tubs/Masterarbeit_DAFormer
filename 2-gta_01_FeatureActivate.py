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
    
data_dir = '/home/yliang/work/DAFormer/data/gta5'

custom_transform = transforms.Compose([    # Data preprocessing and transformation
    transforms.Resize((256, 512)),
    transforms.ToTensor()
])

# Import validation set
with open(os.path.join(data_dir, 'validation.json')) as f:
    validation_samples = json.load(f)

val_image_paths= validation_samples.get('files')[0]
val_image_list = [data_dir + '/' + image for image in val_image_paths]

num_samples_to_select = 500
random_selected_images = random.sample(val_image_list, num_samples_to_select)
# with open('/home/yliang/work/DAFormer/save_file/f_4_gta/500samples/500samples_list.pkl', 'wb') as file:
#     pickle.dump(random_selected_images, file)


with open('/home/yliang/work/DAFormer/save_file/output_feature_new/gta5/500samples_list.pkl', 'rb') as file:
    random_selected_images = pickle.load(file)

# 创建一个新的CustomDataset和Dataloader
custom_dataset_random = CustomDataset(random_selected_images, transform=custom_transform)
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
HRDA_gta = BACKBONES.get(selected_backbone)(init_cfg = dict(type='Pretrained'))
HRDA_gta.load_state_dict(encoder_HRDA_gta)


# 3.Extract features
outputs_baseline = extract_features(model_baseline, dataloader)
outputs_pixmix = extract_features(model_pixmix, dataloader)
outputs_HRDA_gta = extract_features(HRDA_gta, dataloader)

# Save concatenated features
output_dir = '/home/yliang/work/DAFormer/save_file/output_feature_new/gta5'

save_features(outputs_baseline, output_dir ,'baseline', dataset='gta5')
save_features(outputs_pixmix, output_dir , 'pixmix', dataset= 'gta5')
save_features(outputs_HRDA_gta, output_dir , 'HRDA_gta', dataset = 'gta5')


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
    torch.save(concatenated_tensor, f'/home/yliang/work/DAFormer/save_file/before_training_feature/GTA5/f{i+1}_tensor_GTA5.pt')
    print(f'saved: /home/yliang/work/DAFormer/save_file/before_training_feature/GTA5/f{i+1}_tensor_GTA5.pt')

















'''
# 2.import model Baseline
with open('/home/yliang/work/DAFormer/save_file/encoder_baseline.pkl', 'rb') as file:
    encoder_baseline = pickle.load(file)
print('encoder_baseline_loaded')
selected_backbone = 'mit_b5' # import model and set weight
model_baseline  = BACKBONES.get(selected_backbone)()
model_baseline.load_state_dict(encoder_baseline)

# 3. Get Feature Vector
model_baseline.eval()

outputs_f1 = []
outputs_f2 = []
outputs_f3 = []
outputs_f4 = []

print('activating ...')
with torch.no_grad():
    for batch in dataloader:
        print("Processing batch...")
        images = batch
        outputs_batchsize_1 = model_baseline(images)
        outputs_f1.append(outputs_batchsize_1[0])
        outputs_f2.append(outputs_batchsize_1[1])
        outputs_f3.append(outputs_batchsize_1[2])
        outputs_f4.append(outputs_batchsize_1[-1])
        print("Batch processed successfully.")

f_1 = torch.cat(outputs_f1, dim=0) # [500, 64, 64, 128]
f_1_permuted = f_1.permute(0, 2, 3, 1) # [500, 64, 128, 64]

f_2 = torch.cat(outputs_f2, dim=0) # [500, 128, 32, 64]
f_2_permuted = f_2.permute(0, 2, 3, 1) # [500, 32, 64, 128]

f_3 = torch.cat(outputs_f3, dim=0) # [500, 320, 16, 32]
f_3_permuted = f_3.permute(0, 2, 3, 1) # [500, 16, 32, 320]

f_4 = torch.cat(outputs_f4, dim=0) # [500, 512, 8, 16]
f_4_permuted = f_4.permute(0, 2, 3, 1) # [500, 8, 16, 512]

torch.save(f_1_permuted, '/home/yliang/work/DAFormer/save_file/output_feature/gta5/f1/f1_baseline_tensor_gta.pt')
torch.save(f_2_permuted, '/home/yliang/work/DAFormer/save_file/output_feature/gta5/f2/f2_baseline_tensor_gta.pt')
torch.save(f_3_permuted, '/home/yliang/work/DAFormer/save_file/output_feature/gta5/f3/f3_baseline_tensor_gta.pt')
torch.save(f_4_permuted, '/home/yliang/work/DAFormer/save_file/output_feature/gta5/f4/f4_baseline_tensor_gta.pt')





#Pix mix
# 1.import model Pixmix
with open('/home/yliang/work/DAFormer/save_file/encoder_Pixmix.pkl', 'rb') as file:
    encoder_pixmix= pickle.load(file)
print('encoder_pixmix loaded')

selected_backbone = 'mit_b5' # import model and set weight
model_pixmix  = BACKBONES.get(selected_backbone)()
model_pixmix.load_state_dict(encoder_pixmix)

# 2. Get Feature Vector
model_pixmix.eval()

outputs_f1 = []
outputs_f2 = []
outputs_f3 = []
outputs_f4 = []

print('activating ...')
with torch.no_grad():
    for batch in dataloader:
        print("Processing batch...")
        images = batch
        outputs_batchsize_1 = model_baseline(images)
        outputs_f1.append(outputs_batchsize_1[0])
        outputs_f2.append(outputs_batchsize_1[1])
        outputs_f3.append(outputs_batchsize_1[2])
        outputs_f4.append(outputs_batchsize_1[-1])
        print("Batch processed successfully.")

f_1 = torch.cat(outputs_f1, dim=0) # [500, 64, 64, 128]
f_1_permuted = f_1.permute(0, 2, 3, 1) # [500, 64, 128, 64]

f_2 = torch.cat(outputs_f2, dim=0) # [500, 128, 32, 64]
f_2_permuted = f_2.permute(0, 2, 3, 1) # [500, 32, 64, 128]

f_3 = torch.cat(outputs_f3, dim=0) # [500, 320, 16, 32]
f_3_permuted = f_3.permute(0, 2, 3, 1) # [500, 16, 32, 320]

f_4 = torch.cat(outputs_f4, dim=0) # [500, 512, 8, 16]
f_4_permuted = f_4.permute(0, 2, 3, 1) # [500, 8, 16, 512]

torch.save(f_1_permuted, '/home/yliang/work/DAFormer/save_file/output_feature/gta5/f1/f1_pixmix_tensor_gta.pt')
torch.save(f_2_permuted, '/home/yliang/work/DAFormer/save_file/output_feature/gta5/f2/f2_pixmix_tensor_gta.pt')
torch.save(f_3_permuted, '/home/yliang/work/DAFormer/save_file/output_feature/gta5/f3/f3_pixmix_tensor_gta.pt')
torch.save(f_4_permuted, '/home/yliang/work/DAFormer/save_file/output_feature/gta5/f4/f4_pixmix_tensor_gta.pt')
'''
