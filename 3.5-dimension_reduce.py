import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import time

# if torch.cuda.is_available():
#     device = torch.device('cuda')
#     print('CUDA is available. Using GPU...')
# else:
#     device = torch.device('cpu')
#     print('CUDA is not available. Using CPU...')
device = torch.device('cpu')

# cityscapes_labels = labels

def dim_reduce(label_tensor1_file:str,label_tensor2_file:str, activata_feature_tensor1_file:str, activata_feature_tensor2_file:str, max_class=1000):
    """
    Description of the function.

    Parameters:
    label_tensor_file (str): the path of label tensor.
    activata_feature_tensor_file (str): the path of activation feature.
    ...

    Returns:
    selected_indices(list): indices which corresponds to selected pixels.
    pixels_reduced_2d(ndarray): 2 dimensional vectors of selected pixels.
    """
    label_tensor_1 = torch.load(label_tensor1_file, map_location=device)
    label_tensor_2 = torch.load(label_tensor2_file, map_location=device)

    flat_label_1 = label_tensor_1.flatten()
    flat_label_2 = label_tensor_2.flatten()

    selected_indices_1 = []
    selected_indices_2 = []
 
    np.random.seed(42)
    for label in np.unique(flat_label_1):
        indices = np.where(flat_label_1 == label)[0]
        if len(indices) > max_class:
            selected_indices_1.extend(np.random.choice(indices, max_class, replace=False))
        else:
            selected_indices_1.extend(indices)

    np.random.seed(42)
    for label in np.unique(flat_label_2):
        indices = np.where(flat_label_2 == label)[0]
        if len(indices) > max_class:
            selected_indices_2.extend(np.random.choice(indices, max_class, replace=False))
        else:
            selected_indices_2.extend(indices)


    # 从 pixel_data 中选择所选的像素点
    output_1 = torch.load(activata_feature_tensor1_file, map_location=device)
    output_1_reshaped = output_1.reshape(-1, output_1.size()[-1])
    selected_pixel_1 = output_1_reshaped[selected_indices_1]

    output_2 = torch.load(activata_feature_tensor2_file, map_location=device)
    output_2_reshaped = output_2.reshape(-1, output_2.size()[-1])
    selected_pixel_2 = output_2_reshaped[selected_indices_2]

    combined_pixels = torch.cat((selected_pixel_1, selected_pixel_2), dim=0)

    start_time = time.time()

    # 使用 PCA 进行降维
    pca = PCA(n_components=30, random_state=2023)  # 假设降维到30维
    pixels_reduced = pca.fit_transform(combined_pixels)
    print('pca reduced successed')

    # 使用 t-SNE 进行进一步降维到2维
    print('tSNE reducing')
    # tsne = TSNE(n_components=2, random_state=2023, perplexity=40)
    tsne = TSNE(n_components=2, random_state=2023, perplexity=50, n_iter=2000)
    pixels_reduced_2d = tsne.fit_transform(pixels_reduced)
    
    end_time = time.time()
    execution_time = end_time - start_time
    print(execution_time)

    reduced_selected_pixel_1 = torch.from_numpy(pixels_reduced_2d[:len(selected_pixel_1)])
    reduced_selected_pixel_2 = torch.from_numpy(pixels_reduced_2d[len(selected_pixel_1):])

    selected_label_1 = flat_label_1[selected_indices_1]
    selected_label_2 = flat_label_2[selected_indices_2]

 
    # Return statement
    return selected_label_1, selected_label_2, reduced_selected_pixel_1, reduced_selected_pixel_2  # or whatever result you need

dataset ='gta5'     # cityscapes/gta5/synthia
model1 =  'baseline'                         # baseline/pixmix/HRDA_gta
model2='HRDA_gta'
# feature = ['f1','f2','f3','f4']                         # f1/f2/f3/f4
feature = ['f1']

for layer in feature:
    print(f'caculate {layer}')
    label_tensor1_file = f'/home/yliang/work/DAFormer/save_file/output_feature_new/{dataset}/label_Id_tensor/{layer}_labels_tensor_{dataset}.pt'
    label_tensor2_file = f'/home/yliang/work/DAFormer/save_file/output_feature_new/{dataset}/label_Id_tensor/{layer}_labels_tensor_{dataset}.pt'

    activata_feature_tensor1_file =f'/home/yliang/work/DAFormer/save_file/output_feature_new/{dataset}/{layer}/{layer}_{model1}_tensor_{dataset}.pt'
    activata_feature_tensor2_file =f'/home/yliang/work/DAFormer/save_file/output_feature_new/{dataset}/{layer}/{layer}_{model2}_tensor_{dataset}.pt'


    selected_labels_1, selected_labels_2, reduced_selected_pixel_1, reduced_selected_pixel_2= dim_reduce(label_tensor1_file, label_tensor2_file, activata_feature_tensor1_file, activata_feature_tensor2_file)
    
    #save
    torch.save(reduced_selected_pixel_1, f'/home/yliang/work/DAFormer/save_file/output_feature_new/visualize_plot/transparent/{model1}/pixels_labels/{model1}_{layer}_{dataset}_tensor.pt')
    torch.save(reduced_selected_pixel_2, f'/home/yliang/work/DAFormer/save_file/output_feature_new/visualize_plot/transparent/{model2}/pixels_labels/{model2}_{layer}_{dataset}_tensor.pt')
    print('pixels saved')

    np.save(f'/home/yliang/work/DAFormer/save_file/output_feature_new/visualize_plot/transparent/{model1}/pixels_labels/{model1}_{layer}_{dataset}_selected_label.npy', selected_labels_1)
    np.save(f'/home/yliang/work/DAFormer/save_file/output_feature_new/visualize_plot/transparent/{model2}/pixels_labels/{model2}_{layer}_{dataset}_selected_label.npy', selected_labels_2)
    print('labels saved')
