import torch
import matplotlib.pyplot as plt

def load_baseline_weights(model_path, device):
    checkpoint = torch.load(model_path, map_location=device)
    encoder_weights = {}
    for name, param in checkpoint['state_dict'].items():
        if 'backbone' in name:  
            name = name.replace("backbone.", "") 
            encoder_weights[name] = param
    return encoder_weights

def load_hrda_weights(model_path, device):
    checkpoint = torch.load(model_path, map_location=device)
    encoder_weights = {}
    for name, param in checkpoint['state_dict'].items():
        if 'model.backbone' in name and 'ema_' not in name and 'imnet_' not in name:
            name = name.replace("model.backbone.", "") 
            encoder_weights[name] = param
    return encoder_weights

def calculate_cosine_similarity(weights1, weights2):
    cosine_similarities = []
    for key1, value1 in weights1.items():
        if key1 in weights2:
            value2 = weights2[key1]
            similarity = torch.nn.functional.cosine_similarity(value1.flatten().float(), value2.flatten().float(), dim=0)
            cosine_similarities.append((key1, similarity.item()))
    return cosine_similarities

def calculate_mae(weights1, weights2):
    mae_values = []
    for key1, value1 in weights1.items():
        if key1 in weights2:
            value2 = weights2[key1]
            mae = torch.mean(torch.abs(value1.float() - value2.float()))
            mae_values.append((key1, mae.item()))
    return mae_values

def calculate_mse(weights1, weights2):
    mse_values = []
    for key1, value1 in weights1.items():
        if key1 in weights2:
            value2 = weights2[key1]
            mse = torch.mean((value1.float() - value2.float()) ** 2)
            mse_values.append((key1, mse.item()))
    return mse_values

def calculate_dot_product(weights1, weights2):
    dot_product_values = []
    for key1, value1 in weights1.items():
        if key1 in weights2:
            value2 = weights2[key1]
            dot_product = torch.dot(value1.view(-1), value2.view(-1))
            dot_product_values.append((key1, dot_product.item()))
    return dot_product_values

if torch.cuda.is_available():
    device = torch.device('cuda')
    print('CUDA is available. Using GPU...')
else:
    device = torch.device('cpu')
    print('CUDA is not available. Using CPU...')

model1_path = '/home/yliang/work/DAFormer/pretrained/BaselineA.pth'
model2_path = '/home/yliang/work/DAFormer/pretrained/HRDA_GTA.pth'
model3_path = '/home/yliang/work/DAFormer/pretrained/HRDA_Synthia.pth'

weights_baseline = load_baseline_weights(model1_path, device)
weights_hrda_gta = load_hrda_weights(model2_path, device)
weights_hrda_synthia = load_hrda_weights(model3_path, device)

# Define the block structure
block_structure = {}
for i in range(3):  # block1.
    list_name = f"block1.{i}"  
    block_structure[list_name] = []

    for layer, weight in weights_baseline.items():
        if f"block1.{i}" in layer:
            block_structure[list_name].append(layer)

for i in range(6):  # block2.
    list_name = f"block2.{i}"  
    block_structure[list_name] = []

    for layer, weight in weights_baseline.items():
        if f"block2.{i}" in layer:
            block_structure[list_name].append(layer)

for i in range(40):  # block3.
    list_name = f"block3.{i}"  
    block_structure[list_name] = []

    for layer, weight in weights_baseline.items():
        if f"block3.{i}" in layer:
            block_structure[list_name].append(layer)

for i in range(3):  # block4.
    list_name = f"block4.{i}"  
    block_structure[list_name] = []

    for layer, weight in weights_baseline.items():
        if f"block4.{i}" in layer:
            block_structure[list_name].append(layer)


# Calculate cosine similarity, MAE, and MSE
cosine_similarities = calculate_cosine_similarity(weights_baseline, weights_hrda_gta)
mae_values = calculate_mae(weights_baseline, weights_hrda_gta)
mse_values = calculate_mse(weights_baseline, weights_hrda_gta)
dot_product_values = calculate_dot_product(weights_baseline, weights_hrda_gta)



# Group data for plotting
layer_names = [item[0] for item in cosine_similarities]

grouped_layer_names = []
for block_name, block_layers in block_structure.items():
    block_layer_indices = [layer_names.index(layer) for layer in block_layers]
    grouped_layer_names.append(block_name)

grouped_cosine_similarities = {}
for block_name, block_layers in block_structure.items():
    block_layer_indices = [layer_names.index(layer) for layer in block_layers]
    block_similarities = [cosine_similarities[i][1] for i in block_layer_indices]
    avg_similarity = sum(block_similarities) / len(block_similarities)
    grouped_cosine_similarities[block_name] = avg_similarity

grouped_mae_values = {}
for block_name, block_layers in block_structure.items():
    block_layer_indices = [layer_names.index(layer) for layer in block_layers]
    block_maes = [mae_values[i][1] for i in block_layer_indices]
    avg_mae = sum(block_maes) / len(block_maes)
    grouped_mae_values[block_name] = avg_mae

grouped_mse_values = {}
for block_name, block_layers in block_structure.items():
    block_layer_indices = [layer_names.index(layer) for layer in block_layers]
    block_mses = [mse_values[i][1] for i in block_layer_indices]
    avg_mse = sum(block_mses) / len(block_mses)
    grouped_mse_values[block_name] = avg_mse

grouped_dot_product_values = {}
for block_name, block_layers in block_structure.items():
    block_layer_indices = [layer_names.index(layer) for layer in block_layers]
    block_dot_products = [dot_product_values[i][1] for i in block_layer_indices]
    avg_dot_product = sum(block_dot_products) / len(block_dot_products)
    grouped_dot_product_values[block_name] = avg_dot_product


def min_max_normalize(values):
    min_val = min(values)
    max_val = max(values)
    normalized_values = [(val - min_val) / (max_val - min_val) for val in values]
    return normalized_values

# Normalize cosine similarities
normalized_cosine_similarities = min_max_normalize(grouped_cosine_similarities.values())

# Normalize MAE values
normalized_mae_values = min_max_normalize(grouped_mae_values.values())

# Normalize MSE values
normalized_mse_values = min_max_normalize(grouped_mse_values.values())

# Normalize dot product values
normalized_dot_product_values = min_max_normalize(grouped_dot_product_values.values())

# Group data for plotting
metrics_data = {
    'Cosine Similarity': {'block_names': list(grouped_cosine_similarities.keys()), 'avg_values': normalized_cosine_similarities},
    # 'MAE': {'block_names': list(grouped_mae_values.keys()), 'avg_values': normalized_mae_values},
    # 'MSE': {'block_names': list(grouped_mse_values.keys()), 'avg_values': normalized_mse_values},
    # 'Dot Product': {'block_names': list(grouped_dot_product_values.keys()), 'avg_values': normalized_dot_product_values}
}

# Plot metrics
plt.figure(figsize=(10, 6))

# 定义不同指标对应的颜色
colors = {'Cosine Similarity': 'blue', 'MAE': 'green', 'MSE': 'red', 'Dot Product':'skyblue'}

for metric_name, grouped_data in metrics_data.items():
    plt.plot(grouped_data['block_names'], grouped_data['avg_values'], marker='o', linestyle='-',linewidth=2.0, color=colors[metric_name], label=metric_name, zorder=10)



plt.xlabel('Blocks', fontsize=14)
plt.ylabel('Metric Value', fontsize=14)
# plt.title('block average')

# plt.xticks(grouped_data['block_names'][::4], grouped_data['block_names'][::4] ,rotation=90, fontsize=12)
plt.xticks(['block1.0', 'block2.0','block3.0', 'block3.3','block3.31','block3.35','block3.38','block4.0'], 
           ['B1', 'B2', 'B3', 'B3.3', 'B3.31', 'B3.35', 'B3.38','B4'], fontsize=12)  # 设置特定刻度标签
# plt.xticks(['block1.0', 'block2.0','block3.0','block4.0'], 
#            ['B1', 'B2', 'B3', 'B4'], fontsize=12)  # 设置特定刻度标签

plt.yticks(fontsize=14)
plt.ylim(0, 1)  # 设置y轴的范围为0到1
plt.xlim(grouped_data['block_names'][0], grouped_data['block_names'][-1])  # 设置x轴的范围为第一个点到最后一个刻度点


# plt.legend(loc='upper center', ncol=4, fontsize=12, labels=['  ', '  ', '  ', '    '])
plt.grid(color='gray', linestyle='-', linewidth=0.5)
# plt.grid(color='black', linestyle='--', linewidth=0.5, which='both', alpha=0.5)  # 在主刻度和次刻度之间绘制网格线
# plt.grid(color='black', linestyle='--', linewidth=0.2, which='minor', alpha=0.2)  # 设置次刻度的网格线粗细
plt.tight_layout()


plt.savefig('/home/yliang/work/DAFormer/save_file/similarity_comparision_new/0_weight_comparision_plot/final/only_cosine/encoder_weight_similarity_bl_hrda_gta5_extrablock.png',
            dpi=600, bbox_inches='tight', pad_inches=0.1)





# Layers only
def calculate_similarity(weights1, weights2):
    cosine_similarities = []
    mae_values = []
    mse_values = []
    dot_product_values = []
    
    for key1, value1 in weights1.items():
        # if 'patch' in key1 and'norm' in key1 and 'weight' in key1:
        # if 'patch_embed' in key1:
        # if 'attn.norm.weight' in key1 or 'block4.0.attn.proj.weight' in key1 or 'block4.1.attn.proj.weight' in key1 or 'block4.2.attn.proj.weight' in key1:
        # if 'attn.proj.weight' in key1 :
        # if 'norm1.weight' == key1 or 'norm2.weight' == key1 or 'norm3.weight' == key1 or 'norm4.weight' == key1:       
        if 'dwconv.dwconv.weight' in key1:
        # if 'fc1.weight' in key1:
        # if 'fc2.weight' in key1:
            value2 = weights2[key1]
            cos_similarity = torch.nn.functional.cosine_similarity(value1.flatten().float(), value2.flatten().float(), dim=0)
            cosine_similarities.append((key1, cos_similarity.item()))

            mae = torch.mean(torch.abs(value1.float() - value2.float()))
            mae_values.append((key1, mae.item()))
            
            mse = torch.mean((value1.float() - value2.float()) ** 2)
            mse_values.append((key1, mse.item()))
            
            dot_product = torch.dot(value1.view(-1), value2.view(-1))
            dot_product_values.append((key1, dot_product.item()))
            print(key1)
            
    return cosine_similarities, mae_values, mse_values, dot_product_values


cos_patch, mae_patch, mse_patch, dot_patch = calculate_similarity(weights_baseline, weights_hrda_gta)

names = [item[0] for item in cos_patch]

# Normalize data
normalized_cos_patch = min_max_normalize([item[1] for item in cos_patch])
normalized_mae_patch = min_max_normalize([item[1] for item in mae_patch])
normalized_mse_patch = min_max_normalize([item[1] for item in mse_patch])
normalized_dot_patch = min_max_normalize([item[1] for item in dot_patch])



plt.figure(figsize=(10, 6))

# 定义不同指标对应的颜色
colors = {'Cosine Similarity': 'blue', 'MAE': 'green', 'MSE': 'red', 'Dot Product':'skyblue'}

# Plot data
plt.plot([item[0] for item in cos_patch], normalized_cos_patch, marker='o', linestyle='-', linewidth=2.0,color=colors['Cosine Similarity'], label='Cosine Similarity')
plt.plot([item[0] for item in mae_patch], normalized_mae_patch, marker='o', linestyle='-', linewidth=2.0,color=colors['MAE'], label='MAE')
plt.plot([item[0] for item in mse_patch], normalized_mse_patch, marker='o', linestyle='-', linewidth=2.0,color=colors['MSE'], label='MSE')
plt.plot([item[0] for item in dot_patch], normalized_dot_patch, marker='o', linestyle='-', linewidth=2.0,color=colors['Dot Product'], label='Dot Product')

plt.xlabel('Fully Connected Layers', fontsize=14)
plt.ylabel('Metric Value', fontsize=14)



# plt.xticks(['block1.0.mlp.fc1.weight', 'block2.0.mlp.fc1.weight',
#             'block3.0.mlp.fc1.weight', 'block4.0.mlp.fc1.weight'], 
#            ['B1', 'B2', 'B3', 'B4'], fontsize=12)  # 设置特定刻度标签
# plt.xticks(['patch_embed1.proj.weight', 'patch_embed2.proj.weight',
#             'patch_embed3.proj.weight', 'patch_embed4.proj.weight'], 
#            ['B1', 'B2', 'B3', 'B4'], fontsize=12)  # 设置特定刻度标签
# plt.xticks(['block1.0.attn.proj.weight', 'block2.0.attn.proj.weight',
#             'block3.0.attn.proj.weight', 'block4.0.attn.proj.weight'], 
#            ['B1', 'B2', 'B3', 'B4'], fontsize=12)  # 设置特定刻度标签
plt.xticks(['block1.0.mlp.dwconv.dwconv.weight', 'block2.0.mlp.dwconv.dwconv.weight',
            'block3.0.mlp.dwconv.dwconv.weight', 'block4.0.mlp.dwconv.dwconv.weight'], 
           ['B1', 'B2', 'B3', 'B4'], fontsize=12)  # 设置特定刻度标签
# plt.xticks(['block1.0.mlp.fc2.weight', 'block2.0.mlp.fc2.weight',
#             'block3.0.mlp.fc2.weight', 'block4.0.mlp.fc2.weight'], 
#            ['B1', 'B2', 'B3', 'B4'], fontsize=12)  # 设置特定刻度标签

plt.yticks(fontsize=14)
plt.ylim(0, 1)  # 设置y轴的范围为0到1
plt.xlim(names[0], names[-1])  # 设置x轴的范围为第一个点到最后一个刻度点

# plt.title('Normalized Metrics Comparison', fontsize=16)
# plt.legend(loc='center', ncol=4, fontsize=12, labels=['  ', '  ', '  ', '    '])
# plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=4, fontsize=12, labels=['  ', '  ', '  ', '    '])
plt.grid(color='gray', linestyle='-', linewidth=0.5)
plt.tight_layout()
# plt.savefig('/home/yliang/work/DAFormer/save_file/similarity_comparision_new/0_weight_comparision_plot/final/4_mlp.fc2.weight.png', dpi=600, bbox_inches='tight', pad_inches=0.1)
plt.savefig('/home/yliang/work/DAFormer/save_file/similarity_comparision_new/0_weight_comparision_plot/final/3_dwcon.png', dpi=600, bbox_inches='tight', pad_inches=0.1)






# legend
plt.figure(figsize=(2, 1))
# 添加一个空的线条，仅用于显示图例
plt.plot([], [],marker='o', linestyle='-', linewidth=2.0, label='Cosine Similarity', color='blue')
plt.plot([], [],marker='o', linestyle='-',  linewidth=2.0,label='MAE', color='green')
plt.plot([], [],marker='o', linestyle='-',  linewidth=2.0,label='MSE', color='red')
plt.plot([], [],marker='o', linestyle='-', linewidth=2.0, label='Dot Product', color='skyblue')
# 不显示坐标轴
plt.axis('off')
# 显示图例
plt.legend(loc='center', ncol=4, fontsize=16, labels=['    ', '    ', '    ', '      '])
# plt.legend(loc='center',ncol=4, fontsize=12, labels=['    '])
# 保存图例图片
plt.savefig('/home/yliang/work/DAFormer/save_file/similarity_comparision_new/0_weight_comparision_plot/final/legend_only.png', dpi=600, bbox_inches='tight', pad_inches=0.1)
