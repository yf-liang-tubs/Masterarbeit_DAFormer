import torch
import numpy as np
import pickle
# import random
# from torch.utils.data import DataLoader, Dataset
# from torchvision import transforms
# from torchvision.datasets import Cityscapes
# from mmseg.models.builder import SEGMENTORS

from PIL import Image


# if torch.cuda.is_available():
#     device = torch.device('cuda')
#     print('CUDA is available. Using GPU...')
# else:
#     device = torch.device('cpu')
#     print('CUDA is not available. Using CPU...')

device = torch.device('cpu')

'''
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

# Create the Cityscapes dataset
cityscapes_dataset = Cityscapes(data_dir, split='val', mode='fine', target_type='semantic',
                                transform=custom_transform, target_transform=custom_transform)


cs_dataloader = DataLoader(cityscapes_dataset,
                           pin_memory=False,
                        batch_size=batch_size, 
                        shuffle=False,
                        worker_init_fn=seed_worker,
                        generator=g)

# gta and synthia
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

# gta5
with open('/home/yliang/work/DAFormer/save_file/output_feature_new/gta5/500samples_list.pkl', 'rb') as file:
    random_selected_images = pickle.load(file)

# 创建一个新的CustomDataset和Dataloader
custom_dataset_random = CustomDataset(random_selected_images, transform=custom_transform)
gta5_dataloader= DataLoader(custom_dataset_random,pin_memory=True,
                        batch_size=batch_size, 
                        shuffle=False,
                        worker_init_fn=seed_worker,
                        generator=g)

#synthia
with open('/home/yliang/work/DAFormer/save_file/output_feature_new/synthia/500samples_list.pkl', 'rb') as file:
    random_image_paths = pickle.load(file)
# 创建一个新的CustomDataset和Dataloader
custom_dataset_random = CustomDataset(random_image_paths, transform=custom_transform)
synthia_dataloader= DataLoader(custom_dataset_random,pin_memory=True,
                        batch_size=batch_size, 
                        shuffle=False,
                        worker_init_fn=seed_worker,
                        generator=g)



# Load model
# 1. baseline---------------------------------------------------------------
BL_decode_encode = SEGMENTORS.get('EncoderDecoder')(
    backbone = dict(type='mit_b5',init_cfg=dict(type='Pretrained')),
    decode_head= dict(type='SegFormerHead' ,init_cfg=dict(type='Pretrained'),
                          in_channels=[64, 128, 320, 512],  channels=128,  num_classes=19,
    in_index=[0, 1, 2, 3], feature_strides=[4,8,16,32],
    decoder_params=dict(
            embed_dim=768,
            conv_kernel_size=1
        ),
    norm_cfg=dict(
            type='SyncBN',
            # type='BN',
            requires_grad=True
        )
))

baseline_checkpoint_path = '/home/yliang/work/DAFormer/pretrained/BaselineA.pth'
checkpoint = torch.load(baseline_checkpoint_path, map_location=device)


state_dict = checkpoint['state_dict']

BL_decode_encode.load_state_dict(state_dict)
bl_decode_encode=BL_decode_encode.to(device)
bl_decode_encode.eval()

del checkpoint

#Test---------------------------------------------------
# input_data = torch.randn(1, 3, 256, 512).to(device)

# forward dummy
# forward_dummy_result = bl_decode_encode.forward_dummy(input_data)
# torch.save(forward_dummy_result, f'/home/yliang/work/DAFormer/save_file/encode_decode_feature/baseline/forward_dummy_result.pt')
# forward_dummy_result = torch.load('/home/yliang/work/DAFormer/save_file/encode_decode_feature/baseline/forward_dummy_result.pt', map_location=device)





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
    skip_batches = 0
    end_batches = 140
    with torch.no_grad():
        for data in dataloader:
            
            batch_counter += 1
            
            if batch_counter <= skip_batches:
                continue
            
            if batch_counter % 10 == 0:
                print(f"Processed {batch_counter} batches")
            
            if batch_counter > end_batches:
                print(f'break batch:{batch_counter}')
                break
            
            input_data, img_metas= data
            inputs = input_data.to(device)
            # img_mets = img_metas.to(device) #cs
            img_mets = {key: value.to(device) for key, value in img_metas.items()} #gta5
            # model(inputs)
            
            # model(inputs, img_metas=img_metas, gt_semantic_seg=gt_semantic_seg)
            model.encode_decode(inputs,img_mets)
            
            del input_data, img_metas, inputs, img_mets
            

            
    for hook in hooks:
        hook.remove()

    return activations


layer_names = []
for name, module in bl_decode_encode.named_modules():
    # print(name)
    
    # 1.f1,f2,f3,f4: [norm1, norm2...norm4]  we have already
    # 2.patch embedding blocks:[patch_embed1.norm, patch_embed2.norm, patch_embed3.norm, patch_embed4.norm]
    # 3.attention blocks: [block1.0.attn.norm, block1.1.attn.norm...block3.39.attn.norm]
    # 4.mix-ffn blocks: [block1.0.mlp.drop, block1.1.mlp.drop....block4.2.mlp.drop]
    # 5.convolutional layers
    # 6.fully connected layers
    
    if 'linear_fuse.bn' in name:
        layer_names.append(name)

# target_layer = dict(bl_decode_encode.named_modules())['decode_head.linear_fuse.bn']


extracted_features_baseline = activate_and_extract_features(bl_decode_encode, gta5_dataloader, layer_names)

torch.save(extracted_features_baseline, f'/home/yliang/work/DAFormer/save_file/encode_decode_feature/baseline/gta5/4.pt')
print('Saved: extracted_features')
'''



#contacte feature----------------------------------------------------
extracted_features_1 = torch.load('/home/yliang/work/DAFormer/save_file/encode_decode_feature/baseline/gta5/1.pt', map_location=device)
cat_1= extracted_features_1['decode_head.linear_fuse.bn']
del extracted_features_1


extracted_features_2 = torch.load('/home/yliang/work/DAFormer/save_file/encode_decode_feature/baseline/gta5/2.pt', map_location=device)
cat_2= extracted_features_2['decode_head.linear_fuse.bn']
del extracted_features_2

extracted_features_3 = torch.load('/home/yliang/work/DAFormer/save_file/encode_decode_feature/baseline/gta5/3.pt', map_location=device)
cat_3= extracted_features_3['decode_head.linear_fuse.bn']
del extracted_features_3

extracted_features_4 = torch.load('/home/yliang/work/DAFormer/save_file/encode_decode_feature/baseline/gta5/4.pt', map_location=device)
cat_4= extracted_features_4['decode_head.linear_fuse.bn']
del extracted_features_4

merged_list = cat_1 + cat_2 + cat_3 + cat_4

result_tensor_cs = torch.cat(merged_list, dim=0)
concatenated_tensor_cs = result_tensor_cs.permute(0, 2, 3, 1)
torch.save(concatenated_tensor_cs, '/home/yliang/work/DAFormer/save_file/encode_decode_feature/baseline/gta5/gta5_concat_decode_head.concatenated_tensor_batch500.pt')
print('save cs')
