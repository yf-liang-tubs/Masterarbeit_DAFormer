import torch
import numpy as np
import random
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import Cityscapes
from mmseg.models.builder import BACKBONES

from PIL import Image
import pickle
'''
if torch.cuda.is_available():
    device = torch.device('cuda')
    print('CUDA is available. Using GPU...')
else:
    device = torch.device('cpu')
    print('CUDA is not available. Using CPU...')
'''
device = torch.device('cpu')

# 1.activate feature
def load_model_from_checkpoint(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)
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


# Dataloader
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(0)
np.random.seed(0)
random.seed(0)

# Path to the dataset folder
data_dir = '/home/yliang/work/DAFormer/data/cityscapes'

# Data preprocessing and transformation
custom_transform = transforms.Compose([
    transforms.Resize((256, 512)),
    transforms.ToTensor()
])
batch_size = 1
'''
# Cityscapes
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


with open('/home/yliang/work/DAFormer/save_file/output_feature_new/gta5/500samples_list.pkl', 'rb') as file:
    random_selected_images = pickle.load(file)

# 创建一个新的CustomDataset和Dataloader
custom_dataset_random = CustomDataset(random_selected_images, transform=custom_transform)
gta5_dataloader= DataLoader(custom_dataset_random,pin_memory=True,
                        batch_size = batch_size, 
                        shuffle=False,
                        worker_init_fn=seed_worker,
                        generator=g)




# Load model
#Baseline model
baseline_checkpoint_path = '/home/yliang/work/DAFormer/pretrained/BaselineA.pth'
baseline = load_model_from_checkpoint(baseline_checkpoint_path)
baseline = baseline.to(device)  # 将模型移动到CUDA设备上
baseline.eval()


'''
def activate_and_extract_features(model, dataloader, target_layer_names):
    activation = {}
    hooks = []

    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook

    for layer_name in target_layer_names:
        target_layer = dict(model.named_modules())[layer_name]
        hook = target_layer.register_forward_hook(get_activation(layer_name))
        hooks.append(hook)

    batch_counter = 0
    with torch.no_grad():
        for batch in dataloader:
            if batch_counter >= 5:
                break  # 当达到第五个批次后停止传播
            inputs, _ = batch
            
            inputs = inputs.to(device)
            
            model(inputs)
            batch_counter += 1

            # 处理完当前批次后手动释放 GPU 上的变量
            del inputs

    extracted_features = {}
    for layer_name in target_layer_names:
        extracted_features[layer_name] = activation[layer_name]

    for hook in hooks:
        hook.remove()

    return extracted_features
'''

def activate_and_extract_features(model, dataloader, target_layer_names):
    activations = {layer_name: [] for layer_name in target_layer_names}
    hooks = []

    def get_activation(name):
        def hook(model, input, output):
            activations[name].append(output.detach())
        return hook

    for layer_name in target_layer_names:
        target_layer = dict(model.named_modules())[layer_name]
        hook = target_layer.register_forward_hook(get_activation(layer_name))
        hooks.append(hook)

    batch_counter = 0
    with torch.no_grad():
        for batch in dataloader:
            if batch_counter >= 16:
                break
            inputs, _ = batch
            inputs = inputs.to(device)
            model(inputs)
            batch_counter += 1
            if batch_counter % 10 == 0:
                print(f"Processed {batch_counter} batches")
            
            del inputs  # 处理完当前批次后手动释放 GPU 上的变量

    for hook in hooks:
        hook.remove()

    return activations



# layer_names = ['norm3','norm4'] # 设置你想要提取特征的层的名称

layer_names = []
for name, module in baseline.named_modules():
    # 1.f1,f2,f3,f4: [norm1, norm2...norm4]  we have already
    # 2.patch embedding blocks:[patch_embed1.norm, patch_embed2.norm, patch_embed3.norm, patch_embed4.norm]
    # 3.attention blocks: [block1.0.attn.norm, block1.1.attn.norm...block3.39.attn.norm]
    # 4.mix-ffn blocks: [block1.0.mlp.drop, block1.1.mlp.drop....block4.2.mlp.drop]
    # 5.convolutional layers
    # 6.fully connected layers
    
    if ('norm1'== name or 'norm2'==name or 'norm3'==name or 'norm4'==name or 
        'block2.4.mlp.fc2'== name or 'block3.3.mlp.fc2'== name or 
        'block3.31.mlp.fc2'== name or 'block3.35.mlp.fc2'== name
        or 'block3.38.mlp.fc2'== name):
    # if 'patch' in name and'norm' in name:
    # if 'attn.norm' in name or 'block4.0.attn.proj_drop' == name or 'block4.1.attn.proj_drop'== name or 'block4.2.attn.proj_drop' == name:
    # if 'mlp.dwconv.dwconv' in name:
    # if 'fc2' in name:
        layer_names.append(name)
print(layer_names)


extracted_features_baseline = activate_and_extract_features(baseline, gta5_dataloader, layer_names)
torch.save(extracted_features_baseline, '/home/yliang/work/DAFormer/save_file/similarity_comparision_new/1_activation_feature/gta5/0_f1_f2_f3_f4_other_lowest_point/baseline_lowpoints_batch500.pt')
print('Saved: extracted_features')

