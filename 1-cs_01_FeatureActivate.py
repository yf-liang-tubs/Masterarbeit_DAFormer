import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import Cityscapes
from mmseg.models.builder import BACKBONES

# 1. Create Data Loader
data_dir = '/home/yliang/work/DAFormer/data/cityscapes' # Path to the dataset folder

custom_transform = transforms.Compose([    # Data preprocessing and transformation
    transforms.Resize((256, 512)),
    transforms.ToTensor()
])

cityscapes_dataset = Cityscapes(data_dir, split='val', mode='fine', target_type='semantic',
                                transform=custom_transform, target_transform=custom_transform)

dataloader = DataLoader(cityscapes_dataset,
                        batch_size=1,
                        shuffle=False,
                        )



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
            inputs, _ = batch
            outputs_batch = model(inputs)
            for i in range(4):
                outputs[i].append(outputs_batch[i])
            print("Batch processed successfully.")
    return [torch.cat(output_list, dim=0) for output_list in outputs]

def save_features(outputs, output_dir, model, dataset):
    for i, output_list in enumerate(outputs):
        print(output_list.size())
        # concatenated_tensor = torch.cat(output_list, dim=0)
        concatenated_tensor = output_list.permute(0, 2, 3, 1)
        torch.save(concatenated_tensor, f'{output_dir}/f{i+1}/f{i+1}_{model}_tensor_{dataset}.pt')
        print(f'saved: {output_dir}/f{i+1}/f{i+1}_{model}_tensor_{dataset}.pt')


# 2.load model
#Baseline model
baseline_checkpoint_path = '/home/yliang/work/DAFormer/pretrained/BaselineA.pth'
model_baseline = load_model_from_checkpoint(baseline_checkpoint_path)


#Pixmix model
pixmix_checkpoint_path = '/home/yliang/work/DAFormer/pretrained/Pixmix_A.pth'
model_pixmix = load_model_from_checkpoint(pixmix_checkpoint_path)


#HRDA_gta5 model
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



# 3.Extract and features
output_dir = '/home/yliang/work/DAFormer/save_file/output_feature_new/cityscapes'

print('activating cs_bl...')
outputs_baseline = extract_features(model_baseline, dataloader)
save_features(outputs_baseline, output_dir ,'baseline', dataset='cityscapes')

print('activating cs_pixmix...')
outputs_pixmix = extract_features(model_pixmix, dataloader)
save_features(outputs_pixmix, output_dir , 'pixmix', dataset='cityscapes')

print('activating cs_HRDA...')
outputs_HRDA_gta = extract_features(HRDA_gta, dataloader)
save_features(outputs_HRDA_gta, output_dir , 'HRDA_gta', dataset='cityscapes')

outputs_HRDA_synthia = extract_features(HRDA_synthia, dataloader)
save_features(outputs_HRDA_synthia, output_dir , model='HRDA_synthia', dataset='cityscapes')




# 4. before tarining model
before_training_model = BACKBONES.get('mit_b5')(init_cfg = dict(type='Pretrained'))
before_training_model.eval()

outputs = [[] for _ in range(4)]
with torch.no_grad():
    for batch in dataloader:
        inputs, _ = batch
        outputs_batch = before_training_model(inputs)
        for i in range(4):
            outputs[i].append(outputs_batch[i])
        print("Batch processed successfully.")
outputs_cat = [torch.cat(output_list, dim=0) for output_list in outputs]


for i, output_list in enumerate(outputs_cat):
    print(output_list.size())
    # concatenated_tensor = torch.cat(output_list, dim=0)
    concatenated_tensor = output_list.permute(0, 2, 3, 1)
    torch.save(concatenated_tensor, f'/home/yliang/work/DAFormer/save_file/before_training_feature/CS/f{i+1}_tensor_CS.pt')
    print(f'saved: /home/yliang/work/DAFormer/save_file/before_training_feature/CS/f{i+1}_tensor_CS.pt')

# 5.  starkes Synthia trained, adapted model

