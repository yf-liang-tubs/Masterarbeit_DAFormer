import torch
import numpy as np
import random
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import Cityscapes
from similarity_method.dcka import deltaCKA, CKA, deltaRSA, RSA
from similarity_method.rtd.barcodes import rtd,rtd1
import matplotlib.pyplot as plt
from PIL import Image
import pickle


if torch.cuda.is_available():
    device = torch.device('cuda')
    print('CUDA is available. Using GPU...')
else:
    device = torch.device('cpu')
    print('CUDA is not available. Using CPU...')


# Dataloader
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(0)
np.random.seed(0)
random.seed(0)


# Data preprocessing and transformation
custom_transform = transforms.Compose([
    transforms.Resize((256, 512)),
    transforms.ToTensor()
])
batch_size = 20
'''
# Cityscapes
# Path to the dataset folder
data_dir = '/home/yliang/work/DAFormer/data/cityscapes'

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






cka_examples_batch = next(iter(gta5_dataloader)) # retrieves the first batch of data from the cs_dataloader
input_batch, cs_sample_target = cka_examples_batch
input_embedding = input_batch.reshape(input_batch.shape[0], -1)


# 2.import feature
# file_path = '/home/yliang/work/DAFormer/save_file/similarity_comparision/feature_embedding/'
# layers_type = 'attn'

# baseline_norm = torch.load(f'{file_path}baseline_{layers_type}_batch{batch_size}.pt', map_location=device)
# HDRA_norm = torch.load(f'{file_path}HDRA_{layers_type}_batch{batch_size}.pt', map_location=device)
baseline_norm = torch.load(f'/home/yliang/work/DAFormer/save_file/similarity_comparision_new/1_activation_feature/gta5/0_f1_f2_f3_f4/baseline_norm_batch500.pt', map_location=device)
HDRA_norm = torch.load(f'/home/yliang/work/DAFormer/save_file/similarity_comparision_new/1_activation_feature/gta5/0_f1_f2_f3_f4/HDRA_norm_batch500.pt', map_location=device)



cka = CKA(device=device) ; dcka = deltaCKA(device=device)
rsa = RSA(device=device) ; drsa = deltaRSA(device=device)



# 3.Calculate the similarity between layer representations
f1 = baseline_norm['block1.0.attn.norm']
f2 = HDRA_norm['block1.0.attn.norm']

f1 = torch.cat(f1, dim=0)
f2 = torch.cat(f2, dim=0)

cka1 = cka.linear_CKA(f1.reshape(input_batch.shape[0], -1), f2.reshape(input_batch.shape[0], -1))
print('cka = '+ str(round(cka1.item(), 3)))

dcka1 = dcka.linear_CKA(f1.reshape(input_batch.shape[0], -1), f2.reshape(input_batch.shape[0], -1), input_embedding,input_embedding)
print('dcka = ' + str(round(dcka1.item(), 3)))

rsa1 = rsa.compute_rsa(f1.reshape(input_batch.shape[0], -1), f2.reshape(input_batch.shape[0], -1))
print('rsa = '+ str(round(rsa1.item(), 3)))

drsa1 = drsa.compute_deltarsa(f1.reshape(input_batch.shape[0], -1), f2.reshape(input_batch.shape[0], -1), input_embedding,input_embedding)
print('drsa1 = ' + str(round(drsa1.item(), 3)))


rtd1_value = rtd(f1.reshape(input_batch.shape[0], -1), f2.reshape(input_batch.shape[0], -1))
print('rtd1 = '+ str(rtd1_value))




# Plot----------------------------------------------------------------

# 0.f1,f2,f3,f4, and otherr lowerst points--------------------------------
cka_similarities = []
dcka_similarities = []
rsa_similarities = []
drsa_similarities = []

for (key1, value1), (key2, value2) in zip (baseline_norm.items(),HDRA_norm.items()):
        print(key1,key2)
        value1 = torch.cat(value1, dim=0)
        value2 = torch.cat(value2, dim=0)
        cka_similarity = cka.linear_CKA(value1.reshape(input_batch.shape[0], -1), value2.reshape(input_batch.shape[0], -1))
        dcka_similarity = dcka.linear_CKA(value1.reshape(input_batch.shape[0], -1), value2.reshape(input_batch.shape[0], -1), input_embedding,input_embedding)
        rsa_similarity = rsa.compute_rsa(value1.reshape(input_batch.shape[0], -1), value2.reshape(input_batch.shape[0], -1))
        drsa_similarity  = drsa.compute_deltarsa(value1.reshape(input_batch.shape[0], -1), value2.reshape(input_batch.shape[0], -1), input_embedding,input_embedding)
        
        cka_similarities.append((key1, round(cka_similarity.item(), 3)))
        dcka_similarities.append((key1, round(dcka_similarity.item(), 3)))
        rsa_similarities.append((key1, round(rsa_similarity.item(), 3)))
        drsa_similarities.append((key1, round(drsa_similarity.item(), 3)))

keys = [item[0] for item in dcka_similarities]

dcka_values = [item[1] for item in dcka_similarities]
rsa_values = [item[1] for item in rsa_similarities]
drsa_values = [item[1] for item in drsa_similarities]
cka_values = [item[1] for item in cka_similarities]


plt.figure(figsize=(10, 6))

colors = {'CKA': 'skyblue', 'dCKA': 'blue', 'RSA': 'red', 'dRSA':'green'}

plt.plot(keys, cka_values, label='CKA',marker='o', linestyle='-', linewidth=2.0, color=colors['CKA'])
plt.plot(keys, dcka_values, label='dCKA',marker='o', linestyle='-', linewidth=2.0, color=colors['dCKA'])
plt.plot(keys, rsa_values, label='RSA',marker='o', linestyle='-', linewidth=2.0, color=colors['RSA'])
plt.plot(keys, drsa_values, label='dRSA',marker='o', linestyle='-', linewidth=2.0, color=colors['dRSA'])


plt.xlim(keys[0], keys[-1])  # 设置x轴的范围为第一个点到最后一个刻度点
# plt.legend(loc='lower right', ncol=1, framealpha=1.0)

plt.xticks(keys, ['B1', 'B2', 'B2.4','B3','B3.3','B3.31','B3.35','B3.38','B4'], rotation = 0, fontsize=14)

# 设置y轴范围
plt.ylim(ymin=0.67, ymax=1.0)
# 设置y轴刻度显示
plt.yticks([0.7, 0.8, 0.9, 1.0], fontsize=14) 
# plt.yticks(np.arange(0, 1.5, 0.5), fontsize=10)

# plt.title('Patch Embeddings')
plt.xlabel('Layer', fontsize=14)
plt.ylabel('Metric Value', fontsize=14)
plt.grid(color='gray', linestyle='-', linewidth=0.5)

plt.savefig('/home/yliang/work/DAFormer/save_file/similarity_comparision_new/2_feature_comparision_plot/gta5/new/0_norm_and_lowest.png',
            dpi=600, facecolor='w', edgecolor='w', orientation='portrait',
            bbox_inches='tight', pad_inches=0.2, metadata=None)





# 0.f1,f2,f3,f4: [norm1, norm2...norm4]--------------------------------
cka_similarities = []
dcka_similarities = []
rsa_similarities = []
drsa_similarities = []

for (key1, value1), (key2, value2) in zip (baseline_norm.items(),HDRA_norm.items()):
        print(key1,key2)
        value1 = torch.cat(value1, dim=0)
        value2 = torch.cat(value2, dim=0)
        cka_similarity = cka.linear_CKA(value1.reshape(input_batch.shape[0], -1), value2.reshape(input_batch.shape[0], -1))
        dcka_similarity = dcka.linear_CKA(value1.reshape(input_batch.shape[0], -1), value2.reshape(input_batch.shape[0], -1), input_embedding,input_embedding)
        rsa_similarity = rsa.compute_rsa(value1.reshape(input_batch.shape[0], -1), value2.reshape(input_batch.shape[0], -1))
        drsa_similarity  = drsa.compute_deltarsa(value1.reshape(input_batch.shape[0], -1), value2.reshape(input_batch.shape[0], -1), input_embedding,input_embedding)
        
        cka_similarities.append((key1, round(cka_similarity.item(), 3)))
        dcka_similarities.append((key1, round(dcka_similarity.item(), 3)))
        rsa_similarities.append((key1, round(rsa_similarity.item(), 3)))
        drsa_similarities.append((key1, round(drsa_similarity.item(), 3)))

keys = [item[0] for item in dcka_similarities]

dcka_values = [item[1] for item in dcka_similarities]
rsa_values = [item[1] for item in rsa_similarities]
drsa_values = [item[1] for item in drsa_similarities]
cka_values = [item[1] for item in cka_similarities]


plt.figure(figsize=(10, 6))

colors = {'CKA': 'skyblue', 'dCKA': 'blue', 'RSA': 'red', 'dRSA':'green'}

plt.plot(keys, cka_values, label='CKA',marker='o', linestyle='-', linewidth=2.0, color=colors['CKA'])
plt.plot(keys, dcka_values, label='dCKA',marker='o', linestyle='-', linewidth=2.0, color=colors['dCKA'])
plt.plot(keys, rsa_values, label='RSA',marker='o', linestyle='-', linewidth=2.0, color=colors['RSA'])
plt.plot(keys, drsa_values, label='dRSA',marker='o', linestyle='-', linewidth=2.0, color=colors['dRSA'])


plt.xlim(keys[0], keys[-1])  # 设置x轴的范围为第一个点到最后一个刻度点
# plt.legend(loc='lower right', ncol=1, framealpha=1.0)

plt.xticks(keys, ['N1', 'N2', 'N3', 'N4'], rotation = 0, fontsize=14)

# 设置y轴范围
plt.ylim(ymin=0.67, ymax=1.0)
# 设置y轴刻度显示
plt.yticks([0.7, 0.8, 0.9, 1.0], fontsize=14) 
# plt.yticks(np.arange(0, 1.5, 0.5), fontsize=10)

# plt.title('Patch Embeddings')
plt.xlabel('Normalization Layer', fontsize=14)
plt.ylabel('Metric Value', fontsize=14)
plt.grid(color='gray', linestyle='-', linewidth=0.5)

plt.savefig('/home/yliang/work/DAFormer/save_file/similarity_comparision_new/2_feature_comparision_plot/gta5/new/0_norm.png',
            dpi=600, facecolor='w', edgecolor='w', orientation='portrait',
            bbox_inches='tight', pad_inches=0.2, metadata=None)




# 1.patch_embedding-------------------------------------------
cka_similarities = []
dcka_similarities = []
rsa_similarities = []
drsa_similarities = []

for (key1, value1), (key2, value2) in zip (baseline_norm.items(),HDRA_norm.items()):
        print(key1,key2)
        value1 = torch.cat(value1, dim=0)
        value2 = torch.cat(value2, dim=0)
        cka_similarity = cka.linear_CKA(value1.reshape(input_batch.shape[0], -1), value2.reshape(input_batch.shape[0], -1))
        dcka_similarity = dcka.linear_CKA(value1.reshape(input_batch.shape[0], -1), value2.reshape(input_batch.shape[0], -1), input_embedding,input_embedding)
        rsa_similarity = rsa.compute_rsa(value1.reshape(input_batch.shape[0], -1), value2.reshape(input_batch.shape[0], -1))
        drsa_similarity  = drsa.compute_deltarsa(value1.reshape(input_batch.shape[0], -1), value2.reshape(input_batch.shape[0], -1), input_embedding,input_embedding)
        
        cka_similarities.append((key1, round(cka_similarity.item(), 3)))
        dcka_similarities.append((key1, round(dcka_similarity.item(), 3)))
        rsa_similarities.append((key1, round(rsa_similarity.item(), 3)))
        drsa_similarities.append((key1, round(drsa_similarity.item(), 3)))

keys = [item[0] for item in dcka_similarities]
dcka_values = [item[1] for item in dcka_similarities]
rsa_values = [item[1] for item in rsa_similarities]
drsa_values = [item[1] for item in drsa_similarities]
cka_values = [item[1] for item in cka_similarities]

plt.figure(figsize=(10, 6))

colors = {'CKA': 'skyblue', 'dCKA': 'blue', 'RSA': 'red', 'dRSA':'green'}

plt.plot(keys, cka_values, label='CKA',marker='o', linestyle='-', linewidth=2.0, color=colors['CKA'])
plt.plot(keys, dcka_values, label='dCKA',marker='o', linestyle='-', linewidth=2.0, color=colors['dCKA'])
plt.plot(keys, rsa_values, label='RSA',marker='o', linestyle='-', linewidth=2.0, color=colors['RSA'])
plt.plot(keys, drsa_values, label='dRSA',marker='o', linestyle='-', linewidth=2.0, color=colors['dRSA'])


plt.xlim(keys[0], keys[-1])  # 设置x轴的范围为第一个点到最后一个刻度点
# plt.legend(loc='lower right', ncol=1, framealpha=1.0)

plt.xticks(keys, ['P1', 'P2', 'P3', 'P4'], rotation = 0, fontsize=14)


# 设置y轴范围
plt.ylim(ymin=0.67, ymax=1.0)
# 设置y轴刻度显示
plt.yticks([0.7, 0.8, 0.9, 1.0], fontsize=14) 

# plt.title('Patch Embeddings')
plt.xlabel('Patch Embedding Layer', fontsize=14)
plt.ylabel('Metric Value', fontsize=14)
plt.grid(color='gray', linestyle='-', linewidth=0.5)

plt.savefig('/home/yliang/work/DAFormer/save_file/similarity_comparision_new/2_feature_comparision_plot/gta5/new/1_patch_norm.png',
            dpi=600, facecolor='w', edgecolor='w', orientation='portrait',
            bbox_inches='tight', pad_inches=0.2, metadata=None)


# 2.attention block-------------------------------------------
cka_similarities = []
dcka_similarities = []
rsa_similarities = []
drsa_similarities = []

for (key1, value1), (key2, value2) in zip (baseline_norm.items(),HDRA_norm.items()):
        print(key1,key2)
        value1 = torch.cat(value1, dim=0)
        value2 = torch.cat(value2, dim=0)
        cka_similarity = cka.linear_CKA(value1.reshape(input_batch.shape[0], -1), value2.reshape(input_batch.shape[0], -1))
        dcka_similarity = dcka.linear_CKA(value1.reshape(input_batch.shape[0], -1), value2.reshape(input_batch.shape[0], -1), input_embedding,input_embedding)
        rsa_similarity = rsa.compute_rsa(value1.reshape(input_batch.shape[0], -1), value2.reshape(input_batch.shape[0], -1))
        drsa_similarity  = drsa.compute_deltarsa(value1.reshape(input_batch.shape[0], -1), value2.reshape(input_batch.shape[0], -1), input_embedding,input_embedding)
        
        cka_similarities.append((key1, round(cka_similarity.item(), 3)))
        dcka_similarities.append((key1, round(dcka_similarity.item(), 3)))
        rsa_similarities.append((key1, round(rsa_similarity.item(), 3)))
        drsa_similarities.append((key1, round(drsa_similarity.item(), 3)))

keys = [item[0] for item in dcka_similarities]

dcka_values = [item[1] for item in dcka_similarities]
rsa_values = [item[1] for item in rsa_similarities]
drsa_values = [item[1] for item in drsa_similarities]
cka_values = [item[1] for item in cka_similarities]


plt.figure(figsize=(10, 6))

colors = {'CKA': 'skyblue', 'dCKA': 'blue', 'RSA': 'red', 'dRSA':'green'}

plt.plot(keys, cka_values, label='CKA',marker='o', linestyle='-', linewidth=2.0, color=colors['CKA'])
plt.plot(keys, dcka_values, label='dCKA',marker='o', linestyle='-', linewidth=2.0, color=colors['dCKA'])
plt.plot(keys, rsa_values, label='RSA',marker='o', linestyle='-', linewidth=2.0, color=colors['RSA'])
plt.plot(keys, drsa_values, label='dRSA',marker='o', linestyle='-', linewidth=2.0, color=colors['dRSA'])


plt.xlim(keys[0], keys[-1])  # 设置x轴的范围为第一个点到最后一个刻度点
# plt.legend(loc='lower right', ncol=1, framealpha=1.0)

plt.xticks(['block1.0.attn.norm', 'block2.0.attn.norm', 'block3.0.attn.norm', 'block4.0.attn.proj_drop'],
           ['B1', 'B2', 'B3', 'B4'], rotation = 0, fontsize=14)


# 设置y轴范围
plt.ylim(ymin=0.1, ymax=1.0)
# 设置y轴刻度显示
plt.yticks([0.25, 0.50, 0.75, 1.0],['0.25', '0.50', '0.75', 1.0], fontsize=14) 

# plt.title('Patch Embeddings')
plt.xlabel('Attention Block Layer', fontsize=14)
plt.ylabel('Metric Value', fontsize=14)
plt.grid(color='gray', linestyle='-', linewidth=0.5)

plt.savefig('/home/yliang/work/DAFormer/save_file/similarity_comparision_new/2_feature_comparision_plot/gta5/new/2_attn.png',
            dpi=600, facecolor='w', edgecolor='w', orientation='portrait',
            bbox_inches='tight', pad_inches=0.2, metadata=None)





# 3.mix_ffn-------------------------------------------
cka_similarities = []
dcka_similarities = []
rsa_similarities = []
drsa_similarities = []

for (key1, value1), (key2, value2) in zip (baseline_norm.items(),HDRA_norm.items()):
        print(key1,key2)
        value1 = torch.cat(value1, dim=0)
        value2 = torch.cat(value2, dim=0)
        cka_similarity = cka.linear_CKA(value1.reshape(input_batch.shape[0], -1), value2.reshape(input_batch.shape[0], -1))
        dcka_similarity = dcka.linear_CKA(value1.reshape(input_batch.shape[0], -1), value2.reshape(input_batch.shape[0], -1), input_embedding,input_embedding)
        rsa_similarity = rsa.compute_rsa(value1.reshape(input_batch.shape[0], -1), value2.reshape(input_batch.shape[0], -1))
        drsa_similarity  = drsa.compute_deltarsa(value1.reshape(input_batch.shape[0], -1), value2.reshape(input_batch.shape[0], -1), input_embedding,input_embedding)
        
        cka_similarities.append((key1, round(cka_similarity.item(), 3)))
        dcka_similarities.append((key1, round(dcka_similarity.item(), 3)))
        rsa_similarities.append((key1, round(rsa_similarity.item(), 3)))
        drsa_similarities.append((key1, round(drsa_similarity.item(), 3)))

keys = [item[0] for item in dcka_similarities]

dcka_values = [item[1] for item in dcka_similarities]
rsa_values = [item[1] for item in rsa_similarities]
drsa_values = [item[1] for item in drsa_similarities]
cka_values = [item[1] for item in cka_similarities]


plt.figure(figsize=(10, 6))

colors = {'CKA': 'skyblue', 'dCKA': 'blue', 'RSA': 'red', 'dRSA':'green'}

plt.plot(keys, cka_values, label='CKA',marker='o', linestyle='-', color=colors['CKA'])
plt.plot(keys, dcka_values, label='dCKA',marker='o', linestyle='-', color=colors['dCKA'])
plt.plot(keys, rsa_values, label='RSA',marker='o', linestyle='-', color=colors['RSA'])
plt.plot(keys, drsa_values, label='dRSA',marker='o', linestyle='-', color=colors['dRSA'])


plt.xlim(keys[0], keys[-1])  # 设置x轴的范围为第一个点到最后一个刻度点
# plt.legend(loc='lower right', ncol=1, framealpha=1.0)

plt.xticks(['block1.0.mlp.drop', 'block2.0.mlp.drop', 'block3.0.mlp.drop', 'block4.0.mlp.drop'],
           ['B1', 'B2', 'B3', 'B4'], rotation = 0, fontsize=14)


# 设置y轴范围
plt.ylim(ymin=0.1, ymax=1.0)
# 设置y轴刻度显示
plt.yticks([0.25, 0.50, 0.75, 1.0],['0.25', '0.50', '0.75', 1.0], fontsize=14) 

# plt.title('Patch Embeddings')
plt.xlabel('Mix-FFN Block', fontsize=14)
plt.ylabel('Metric Value', fontsize=14)
plt.grid(color='gray', linestyle='-', linewidth=0.5)

plt.savefig('/home/yliang/work/DAFormer/save_file/similarity_comparision_new/2_feature_comparision_plot/gta5/3_mix_ffn.png',
            dpi=600, facecolor='w', edgecolor='w', orientation='portrait',
            bbox_inches='tight', pad_inches=0.2, metadata=None)





# 4.convolutional layers-------------------------------------------
cka_similarities = []
dcka_similarities = []
rsa_similarities = []
drsa_similarities = []

for (key1, value1), (key2, value2) in zip (baseline_norm.items(),HDRA_norm.items()):
        print(key1,key2)
        value1 = torch.cat(value1, dim=0)
        value2 = torch.cat(value2, dim=0)
        cka_similarity = cka.linear_CKA(value1.reshape(input_batch.shape[0], -1), value2.reshape(input_batch.shape[0], -1))
        dcka_similarity = dcka.linear_CKA(value1.reshape(input_batch.shape[0], -1), value2.reshape(input_batch.shape[0], -1), input_embedding,input_embedding)
        rsa_similarity = rsa.compute_rsa(value1.reshape(input_batch.shape[0], -1), value2.reshape(input_batch.shape[0], -1))
        drsa_similarity  = drsa.compute_deltarsa(value1.reshape(input_batch.shape[0], -1), value2.reshape(input_batch.shape[0], -1), input_embedding,input_embedding)
        
        cka_similarities.append((key1, round(cka_similarity.item(), 3)))
        dcka_similarities.append((key1, round(dcka_similarity.item(), 3)))
        rsa_similarities.append((key1, round(rsa_similarity.item(), 3)))
        drsa_similarities.append((key1, round(drsa_similarity.item(), 3)))

keys = [item[0] for item in dcka_similarities]
# cleaned_keys = [key.replace('mlp.dwconv.dwconv', 'conv') for key in keys]
# cleaned_keys = [cleaned_key.replace('block', 'B') for cleaned_key in cleaned_keys]

dcka_values = [item[1] for item in dcka_similarities]
rsa_values = [item[1] for item in rsa_similarities]
drsa_values = [item[1] for item in drsa_similarities]
cka_values = [item[1] for item in cka_similarities]


plt.figure(figsize=(10, 6))

colors = {'CKA': 'skyblue', 'dCKA': 'blue', 'RSA': 'red', 'dRSA':'green'}

plt.plot(keys, cka_values, label='CKA',marker='o', linestyle='-', linewidth=2.0, color=colors['CKA'])
plt.plot(keys, dcka_values, label='dCKA',marker='o', linestyle='-', linewidth=2.0,color=colors['dCKA'])
plt.plot(keys, rsa_values, label='RSA',marker='o', linestyle='-', linewidth=2.0,color=colors['RSA'])
plt.plot(keys, drsa_values, label='dRSA',marker='o', linestyle='-', linewidth=2.0,color=colors['dRSA'])


plt.xlim(keys[0], keys[-1])  # 设置x轴的范围为第一个点到最后一个刻度点
# plt.legend(loc='lower right', ncol=1, framealpha=1.0)

plt.xticks(['block1.0.mlp.dwconv.dwconv', 'block2.0.mlp.dwconv.dwconv', 'block3.0.mlp.dwconv.dwconv', 'block4.0.mlp.dwconv.dwconv'],
           ['B1', 'B2', 'B3', 'B4'], rotation = 0, fontsize=14)

# 设置y轴范围
plt.ylim(ymin=0.75, ymax=1.0)
# 设置y轴刻度显示
plt.yticks([0.8, 0.9, 1.0], fontsize=14) 
# plt.yticks(np.arange(0, 1.5, 0.5), fontsize=10)

# plt.title('Patch Embeddings')
plt.xlabel('Depth-wise Convolution Layer', fontsize=14)
plt.ylabel('Metric Value', fontsize=14)
plt.grid(color='gray', linestyle='-', linewidth=0.5)

plt.savefig('/home/yliang/work/DAFormer/save_file/similarity_comparision_new/2_feature_comparision_plot/gta5/new/3_dwconv.png',
            dpi=600, facecolor='w', edgecolor='w', orientation='portrait',
            bbox_inches='tight', pad_inches=0.2, metadata=None)
 
 
# 5.fully connected layers-------------------------------------------
cka_similarities = []
dcka_similarities = []
rsa_similarities = []
drsa_similarities = []

for (key1, value1), (key2, value2) in zip (baseline_norm.items(),HDRA_norm.items()):
        print(key1,key2)
        value1 = torch.cat(value1, dim=0)
        value2 = torch.cat(value2, dim=0)
        cka_similarity = cka.linear_CKA(value1.reshape(input_batch.shape[0], -1), value2.reshape(input_batch.shape[0], -1))
        dcka_similarity = dcka.linear_CKA(value1.reshape(input_batch.shape[0], -1), value2.reshape(input_batch.shape[0], -1), input_embedding,input_embedding)
        rsa_similarity = rsa.compute_rsa(value1.reshape(input_batch.shape[0], -1), value2.reshape(input_batch.shape[0], -1))
        drsa_similarity  = drsa.compute_deltarsa(value1.reshape(input_batch.shape[0], -1), value2.reshape(input_batch.shape[0], -1), input_embedding,input_embedding)
        
        cka_similarities.append((key1, round(cka_similarity.item(), 3)))
        dcka_similarities.append((key1, round(dcka_similarity.item(), 3)))
        rsa_similarities.append((key1, round(rsa_similarity.item(), 3)))
        drsa_similarities.append((key1, round(drsa_similarity.item(), 3)))

keys = [item[0] for item in dcka_similarities]
# cleaned_keys = [key.replace('.mlp', '') for key in keys]

dcka_values = [item[1] for item in dcka_similarities]
rsa_values = [item[1] for item in rsa_similarities]
drsa_values = [item[1] for item in drsa_similarities]
cka_values = [item[1] for item in cka_similarities]



# fig_width = 433.62001 / 72.27 #1pt is 1/72.27 inches
# fig_height = fig_width / 1.618
# plt.figure(figsize=(fig_width,fig_height), dpi=600)
plt.figure(figsize=(10, 6))

colors = {'CKA': 'skyblue', 'dCKA': 'blue', 'RSA': 'red', 'dRSA':'green'}

plt.plot(keys, cka_values, label='CKA',marker='o', linestyle='-', linewidth=2.0, color=colors['CKA'])
plt.plot(keys, dcka_values, label='dCKA',marker='o', linestyle='-', linewidth=2.0, color=colors['dCKA'])
plt.plot(keys, rsa_values, label='RSA',marker='o', linestyle='-', linewidth=2.0, color=colors['RSA'])
plt.plot(keys, drsa_values, label='dRSA',marker='o', linestyle='-', linewidth=2.0, color=colors['dRSA'])

# plt.legend(loc='lower right', ncol=1, framealpha=1.0)

# plt.xticks([1, 2, 4, 8, 16, 32])
# plt.xlim(xmin=1, xmax=32)
# plt.xticks(keys[::4], cleaned_keys[::4], rotation = 90, fontsize=6)
plt.xlim(keys[0], keys[-1])  # 设置x轴的范围为第一个点到最后一个刻度点

plt.xticks(['block1.0.mlp.fc1', 'block2.0.mlp.fc1',
            'block3.0.mlp.fc1', 'block4.0.mlp.fc1'], 
           ['B1', 'B2', 'B3', 'B4'], fontsize=14)  # 设置特定刻度标签

# 设置y轴范围
plt.ylim(ymin=0.2, ymax=1.0)
# 设置y轴刻度显示
plt.yticks([0.25, 0.50, 0.75, 1.0],['0.25', '0.50', '0.75', '1.0'], fontsize=14) 

# plt.title('Fully connected layers')
plt.xlabel('Fully Connected Layer', fontsize=14)
plt.ylabel('Metric Value', fontsize=14)
plt.grid(color='gray', linestyle='-', linewidth=0.5)

plt.savefig('/home/yliang/work/DAFormer/save_file/similarity_comparision_new/2_feature_comparision_plot/gta5/new/5_mlp_fc1.png',
            dpi=600, facecolor='w', edgecolor='w', orientation='portrait',
            bbox_inches='tight', pad_inches=0.2, metadata=None)
 
 
 
 
 
 
 
# legend
plt.figure(figsize=(2, 1))
# 添加一个空的线条，仅用于显示图例
plt.plot([], [],marker='o', linestyle='-', linewidth=2.0,  label='Cosine Similarity', color='skyblue')
plt.plot([], [],marker='o', linestyle='-', linewidth=2.0, label='MAE', color='blue')
plt.plot([], [],marker='o', linestyle='-', linewidth=2.0, label='MSE', color='red')
plt.plot([], [],marker='o', linestyle='-', linewidth=2.0, label='Dot Product', color='green')
# 不显示坐标轴
plt.axis('off')
# 显示图例
plt.legend(loc='center', ncol=4, fontsize=16, labels=['   ', '   ', '   ', '        '])
# plt.legend(loc='center',ncol=4, fontsize=12, labels=['    '])
# 保存图例图片
plt.savefig('/home/yliang/work/DAFormer/save_file/similarity_comparision_new/2_feature_comparision_plot/gta5/new/legend_only.png', dpi=600, bbox_inches='tight', pad_inches=0.1)

 
 
 
 
 
 

'''
from similarity_method.LSim_GBS.metric.GBS import LSim
gbs= LSim(f1.reshape(input_batch.shape[0], -1), f2.reshape(input_batch.shape[0], -1))
print(gbs)



f1=torch.load('/home/yliang/work/DAFormer/save_file/output_feature_new/cityscapes/f4/f4_baseline_tensor_cityscapes.pt', map_location=device)
f2=torch.load('/home/yliang/work/DAFormer/save_file/output_feature_new/cityscapes/f4/f4_HRDA_gta_tensor_cityscapes.pt', map_location=device)
cka = CKA(device=device) ; dcka = deltaCKA(device=device)
cka1 = cka.linear_CKA(f1.reshape(500, -1), f2.reshape(500, -1))
print(round(cka1.item(), 3))


rtd_value = rtd(f1.reshape(500, -1), f2.reshape(500, -1))
print(rtd_value)
'''



'''
# 1. Dataloader
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

# Create the Cityscapes dataset
cityscapes_dataset = Cityscapes(data_dir, split='val', mode='fine', target_type='semantic',
                                transform=custom_transform, target_transform=custom_transform)

import itertools
num_images = 16
limited_dataset = itertools.islice(cityscapes_dataset, num_images)
limited_list = list(limited_dataset)

batch_size = 1
cs_dataloader = DataLoader(limited_list,
                        batch_size=batch_size, 
                        shuffle=False,
                        worker_init_fn=seed_worker,
                        generator=g)

cka_examples_batch = next(iter(cs_dataloader))
input_batch, cs_sample_target = cka_examples_batch
input_embedding = input_batch.reshape(input_batch.shape[-1], -1)


cs_dataloader = DataLoader(batch_size=1, 
                        shuffle=False,
                        worker_init_fn=seed_worker,
                        generator=g)
'''
