import torch
import matplotlib.pyplot as plt
import math
import numpy as np

if torch.cuda.is_available():
    device = torch.device('cuda')
    print('CUDA is available. Using GPU...')
else:
    device = torch.device('cpu')
    print('CUDA is not available. Using CPU...')
#----------------------------------------------------------------

# 0. Extract two weight matrices
model1_path = '/home/yliang/work/DAFormer/pretrained/BaselineA.pth'
checkpoint_baseline = torch.load(model1_path, map_location=device)
weight_dict_baseline = checkpoint_baseline['state_dict']
encoder_baseline_weights = {}    # Extract encoder's weights
for name, param in checkpoint_baseline['state_dict'].items():
    if 'backbone' in name:  # encoder
        name = name.replace("backbone.","") 
        encoder_baseline_weights[name] = param


# model2_path = '/home/yliang/work/DAFormer/pretrained/Pixmix_A.pth'
# checkpoint_Pixmix_A = torch.load(model2_path, map_location=device)
# weight_dict_Pixmix_A_ = checkpoint_Pixmix_A['state_dict']

model3_path = '/home/yliang/work/DAFormer/pretrained/HRDA_GTA.pth'
HRDA_gta= torch.load(model3_path, map_location=device)
encoder_HRDA_gta={}
for encoder, weight in HRDA_gta['state_dict'].items():
    if 'model.backbone' in encoder and 'ema_' not in encoder and 'imnet_' not in encoder:
        # print(encoder)
        encoder = encoder.replace("model.backbone.","") 
        weight = weight.float()
        encoder_HRDA_gta[encoder] = weight



model4_path = '/home/yliang/work/DAFormer/pretrained/HRDA_Synthia.pth'
HRDA_Synthia= torch.load(model4_path, map_location=device)
encoder_HRDA_Synthia={}
for encoder, weight in HRDA_Synthia['state_dict'].items():
    if 'model.backbone' in encoder and 'ema_' not in encoder and 'imnet_' not in encoder:
        # print(encoder)
        encoder = encoder.replace("model.backbone.","") 
        weight = weight.float()
        HRDA_Synthia[encoder] = weight

# for key1, value1 in weight_dict_baseline.items():
#     print(str(key1)+'->'+str(value1))




# Define the block structure
block_structure = {}

for i in range(3):  # block1.
    list_name = f"block1.{i}"  
    block_structure[list_name] = []

    for layer, weight in encoder_baseline_weights.items():
        if f"block1.{i}" in layer:
            block_structure[list_name].append(layer)

for i in range(6):  # block2.
    list_name = f"block2.{i}"  
    block_structure[list_name] = []

    for layer, weight in encoder_baseline_weights.items():
        if f"block2.{i}" in layer:
            block_structure[list_name].append(layer)

for i in range(40):  # block3.
    list_name = f"block3.{i}"  
    block_structure[list_name] = []

    for layer, weight in encoder_baseline_weights.items():
        if f"block3.{i}" in layer:
            block_structure[list_name].append(layer)

for i in range(3):  # block4.
    list_name = f"block4.{i}"  
    block_structure[list_name] = []

    for layer, weight in encoder_baseline_weights.items():
        if f"block4.{i}" in layer:
            block_structure[list_name].append(layer)



#----------------------------------------------------------------

# 1. Calculate cosine similarity
cosine_similarities = []
for key1, value1 in encoder_baseline_weights.items():
    if key1 in encoder_HRDA_gta:
        value2 = encoder_HRDA_gta[key1]
        similarity = torch.nn.functional.cosine_similarity(value1.flatten().float(), value2.flatten().float(), dim=0)
        cosine_similarities.append((key1, similarity.item()))


# 1.1 Visualization
layer_names = [item[0] for item in cosine_similarities]
similarities = [item[1] for item in cosine_similarities]


# 1.2 Calculate the average similarity within each block
grouped_layer_names = []
grouped_similarities = []

for block_name, block_layers in block_structure.items():
    block_layer_indices = [layer_names.index(layer) for layer in block_layers]
    block_similarities = [similarities[i] for i in block_layer_indices]
    avg_similarity = sum(block_similarities) / len(block_similarities)

    grouped_layer_names.append(block_name)
    grouped_similarities.append(avg_similarity)


# 1.3 Plot
plt.figure(figsize=(10, 6), dpi=600)

plt.plot(grouped_layer_names, grouped_similarities, marker='o', linestyle='-', color='b')

plt.xticks(grouped_layer_names[::4], grouped_layer_names[::4] ,rotation=90, fontsize=6)

plt.xlabel('Blocks')
plt.ylabel('Average Cosine Similarity')

plt.grid(color='black', linestyle='--', linewidth=0.5)

plt.title('Average Cosine Similarity Between BL and DIDEX w/ HRDA')



# 1.4 Save the image
plt.savefig('/home/yliang/work/DAFormer/save_file/similarity_comparision/weight_sim/cosine_similarity_bl_hrda_gta5.png')



#----------------------------------------------------------------

# 2. calculete MAE
mae_values  = []
for key1, value1 in encoder_baseline_weights.items():
    if key1 in encoder_HRDA_gta:
        value2 = encoder_HRDA_gta[key1]
        mae = torch.mean(torch.abs(value1.float() - value2.float()))  # Convert to float
        mae_values.append((key1, mae.item()))
max(mae_values)
min(mae_values)


layer_names = [item[0] for item in mae_values]
mae_scores = [item[1] for item in mae_values]


grouped_layer_names = []
grouped_mae = []

for block_name, block_layers in block_structure.items():
    block_layer_indices = [layer_names.index(layer) for layer in block_layers]
    block_similarities = [mae_scores[i] for i in block_layer_indices]
    avg_mae = sum(block_similarities) / len(block_similarities)

    grouped_layer_names.append(block_name)
    grouped_mae.append(avg_mae)

plt.figure(figsize=(10, 6))
plt.plot(grouped_layer_names, grouped_mae, marker='o', linestyle='-', color='b')
plt.xlabel('Blocks')
plt.ylabel('Mean Absolute Error (MAE)')
plt.title('Mean Absolute Error Between baselineA and Pixmix_A (Grouped by Blocks)')
plt.yscale('log') # Adjust the y-axis scale to logarithmic scale
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig('/home/yliang/work/DAFormer/save_file/mae_bl_hrda_gta5.png')


#----------------------------------------------------------------

# 3. calculete MSE
mse_values  = []
for key1, value1 in encoder_baseline_weights.items():
    if key1 in encoder_HRDA_gta:
        value2 = encoder_HRDA_gta[key1]
        mse = torch.mean((value1.float() - value2.float()) ** 2)
        mse_values.append((key1, mse.item()))
max(mse_values)
min(mse_values)


layer_names = [item[0] for item in mse_values]
mse_scores = [item[1] for item in mse_values]


grouped_layer_names = []
grouped_mse = []

for block_name, block_layers in block_structure.items():
    block_layer_indices = [layer_names.index(layer) for layer in block_layers]
    block_similarities = [mse_scores[i] for i in block_layer_indices]
    avg_mse = sum(block_similarities) / len(block_similarities)

    grouped_layer_names.append(block_name)
    grouped_mse.append(avg_mse)

plt.figure(figsize=(10, 6))
plt.plot(grouped_layer_names, grouped_mse, marker='o', linestyle='-', color='b')
plt.xlabel('Blocks')
plt.ylabel('Mean Squared Error (MAE)')
plt.title('Mean Squared Error Between baselineA and Pixmix_A (Grouped by Blocks)')
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig('/home/yliang/work/DAFormer/save_file/mse_plot_101001.png')



#----------------------------------------------------------------

# 4. calculete Dot Product
dot_product  = []
for key1, value1 in encoder_baseline_weights.items():
    if key1 in encoder_HRDA_gta:
        value2 = encoder_HRDA_gta[key1]
        flat_value1 = value1.view(-1)
        flat_value2 = value2.view(-1)
        dot = torch.dot(flat_value1.clone().detach(), flat_value2.clone().detach())
        dot_product.append((key1, dot.item()))


layer_names = [item[0] for item in dot_product]
dot_product_values = [item[1] for item in dot_product]

grouped_layer_names = []
grouped_dot = []

for block_name, block_layers in block_structure.items():
    block_layer_indices = [layer_names.index(layer) for layer in block_layers]
    block_similarities = [dot_product_values[i] for i in block_layer_indices]
    avg_dot = sum(block_similarities) / len(block_similarities)

    grouped_layer_names.append(block_name)
    grouped_dot.append(avg_dot)

max(grouped_dot)
min(grouped_dot)


plt.figure(figsize=(10, 6))
plt.plot(grouped_layer_names, grouped_dot, marker='o', linestyle='-', color='b')
plt.xlabel('Blocks')
plt.ylabel('Dot Product')
plt.title('Dot Product Between baselineA and Pixmix_A (Grouped by Blocks)')
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig('/home/yliang/work/DAFormer/save_file/dot_plot_101001.png')
