import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from functools import partial
from warnings import warn
from typing import List, Dict
import matplotlib.pyplot as plt
from mpl_toolkits import axes_grid1
import matplotlib.pyplot as plt
import numpy as np
import random


import torchvision.transforms as transforms
from mmseg.models.builder import BACKBONES
from torchvision.datasets import Cityscapes

def add_colorbar(im, aspect=10, pad_fraction=0.5, **kwargs):
    """Add a vertical color bar to an image plot."""
    divider = axes_grid1.make_axes_locatable(im.axes)
    width = axes_grid1.axes_size.AxesY(im.axes, aspect=1./aspect)
    pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
    current_ax = plt.gca()
    cax = divider.append_axes("right", size=width, pad=pad)
    plt.sca(current_ax)
    return im.axes.figure.colorbar(im, cax=cax, **kwargs)


class CKA:
    def __init__(self,
                    model1: nn.Module,
                    model2: nn.Module,
                    model1_name: str = None,
                    model2_name: str = None,
                    model1_layers: List[str] = None,
                    model2_layers: List[str] = None,
                    device: str ='cpu'):
        """

        :param model1: (nn.Module) Neural Network 1
        :param model2: (nn.Module) Neural Network 2
        :param model1_name: (str) Name of model 1
        :param model2_name: (str) Name of model 2
        :param model1_layers: (List) List of layers to extract features from
        :param model2_layers: (List) List of layers to extract features from
        :param device: Device to run the model
        """

        self.model1 = model1
        self.model2 = model2

        self.device = device

        self.model1_info = {}
        self.model2_info = {}

        if model1_name is None:
            self.model1_info['Name'] = model1.__repr__().split('(')[0]
        else:
            self.model1_info['Name'] = model1_name

        if model2_name is None:
            self.model2_info['Name'] = model2.__repr__().split('(')[0]
        else:
            self.model2_info['Name'] = model2_name

        if self.model1_info['Name'] == self.model2_info['Name']:
            warn(f"Both model have identical names - {self.model2_info['Name']}. " \
                    "It may cause confusion when interpreting the results. " \
                    "Consider giving unique names to the models :)")

        self.model1_info['Layers'] = []
        self.model2_info['Layers'] = []

        self.model1_features = {}
        self.model2_features = {}

        if len(list(model1.modules())) > 150 and model1_layers is None:
            warn("Model 1 seems to have a lot of layers. " \
                    "Consider giving a list of layers whose features you are concerned with " \
                    "through the 'model1_layers' parameter. Your CPU/GPU will thank you :)")

        self.model1_layers = model1_layers

        if len(list(model2.modules())) > 150 and model2_layers is None:
            warn("Model 2 seems to have a lot of layers. " \
                    "Consider giving a list of layers whose features you are concerned with " \
                    "through the 'model2_layers' parameter. Your CPU/GPU will thank you :)")

        self.model2_layers = model2_layers

        self._insert_hooks()
        self.model1 = self.model1.to(self.device)
        self.model2 = self.model2.to(self.device)

        self.model1.eval()
        self.model2.eval()

    def _log_layer(self,
                    model: str,
                    name: str,
                    layer: nn.Module,
                    inp: torch.Tensor,
                    out: torch.Tensor):

        if model == "model1":
            self.model1_features[name] = out

        elif model == "model2":
            self.model2_features[name] = out

        else:
            raise RuntimeError("Unknown model name for _log_layer.")

    def _insert_hooks(self):
        # Model 1
        for name, layer in self.model1.named_modules():
            if self.model1_layers is not None:
                if name in self.model1_layers:
                    self.model1_info['Layers'] += [name]
                    layer.register_forward_hook(partial(self._log_layer, "model1", name))
            else:
                self.model1_info['Layers'] += [name]
                layer.register_forward_hook(partial(self._log_layer, "model1", name))

        # Model 2
        for name, layer in self.model2.named_modules():
            if self.model2_layers is not None:
                if name in self.model2_layers:
                    self.model2_info['Layers'] += [name]
                    layer.register_forward_hook(partial(self._log_layer, "model2", name))
            else:

                self.model2_info['Layers'] += [name]
                layer.register_forward_hook(partial(self._log_layer, "model2", name))

    def _HSIC(self, K, L):
        """
        Computes the unbiased estimate of HSIC metric.

        Reference: https://arxiv.org/pdf/2010.15327.pdf Eq (3)
        """
        N = K.shape[0]
        ones = torch.ones(N, 1).to(self.device)
        result = torch.trace(K @ L)
        result += ((ones.t() @ K @ ones @ ones.t() @ L @ ones) / ((N - 1) * (N - 2))).item()
        result -= ((ones.t() @ K @ L @ ones) * 2 / (N - 2)).item()
        return (1 / (N * (N - 3)) * result).item()

    def compare(self,
                dataloader1: DataLoader,
                dataloader2: DataLoader = None) -> None:
        """
        Computes the feature similarity between the models on the
        given datasets.
        :param dataloader1: (DataLoader)
        :param dataloader2: (DataLoader) If given, model 2 will run on this
                            dataset. (default = None)
        """

        if dataloader2 is None:
            warn("Dataloader for Model 2 is not given. Using the same dataloader for both models.")
            dataloader2 = dataloader1

        self.model1_info['Dataset'] = dataloader1.dataset.__repr__().split('\n')[0]
        self.model2_info['Dataset'] = dataloader2.dataset.__repr__().split('\n')[0]

        N = len(self.model1_layers) if self.model1_layers is not None else len(list(self.model1.modules()))
        M = len(self.model2_layers) if self.model2_layers is not None else len(list(self.model2.modules()))

        self.hsic_matrix = torch.zeros(N, M, 3)

        num_batches = min(len(dataloader1), len(dataloader1))

        for (x1, *_), (x2, *_) in tqdm(zip(dataloader1, dataloader2), desc="| Comparing features |", total=num_batches):

            self.model1_features = {}
            self.model2_features = {}
            _ = self.model1(x1.to(self.device))
            _ = self.model2(x2.to(self.device))

            for i, (name1, feat1) in enumerate(self.model1_features.items()):
                X = feat1.flatten(1)
                K = X @ X.t()
                K.fill_diagonal_(0.0)
                self.hsic_matrix[i, :, 0] += self._HSIC(K, K) / num_batches

                for j, (name2, feat2) in enumerate(self.model2_features.items()):
                    Y = feat2.flatten(1)
                    L = Y @ Y.t()
                    L.fill_diagonal_(0)
                    assert K.shape == L.shape, f"Feature shape mistach! {K.shape}, {L.shape}"

                    self.hsic_matrix[i, j, 1] += self._HSIC(K, L) / num_batches
                    self.hsic_matrix[i, j, 2] += self._HSIC(L, L) / num_batches
        
        
        self.hsic_matrix = self.hsic_matrix[:, :, 1] / (self.hsic_matrix[:, :, 0].sqrt() *
                                                        self.hsic_matrix[:, :, 2].sqrt())

        assert not torch.isnan(self.hsic_matrix).any(), "HSIC computation resulted in NANs"

    def export(self) -> Dict:
        """
        Exports the CKA data along with the respective model layer names.
        :return:
        """
        return {
            "model1_name": self.model1_info['Name'],
            "model2_name": self.model2_info['Name'],
            "CKA": self.hsic_matrix,
            "model1_layers": self.model1_info['Layers'],
            "model2_layers": self.model2_info['Layers'],
            "dataset1_name": self.model1_info['Dataset'],
            "dataset2_name": self.model2_info['Dataset'],

        }

    def plot_results(self,
                        save_path: str = None,
                        title: str = None):
        fig, ax = plt.subplots()
        im = ax.imshow(self.hsic_matrix, origin='lower', cmap='magma')
        ax.set_xlabel(f"Layers {self.model2_info['Name']}", fontsize=15)
        ax.set_ylabel(f"Layers {self.model1_info['Name']}", fontsize=15)

        if title is not None:
            ax.set_title(f"{title}", fontsize=18)
        else:
            ax.set_title(f"{self.model1_info['Name']} vs {self.model2_info['Name']}", fontsize=18)

        add_colorbar(im)
        plt.tight_layout()

        if save_path is not None:
            plt.savefig(save_path, dpi=300)

        plt.show()


device = 'cuda'

# 1. Import and clean checkpoint
checkpoint1_path = '/home/yliang/work/DAFormer/pretrained/BaselineA.pth'
checkpoint_baseline = torch.load(checkpoint1_path, map_location=torch.device(device))
encoder_baseline={}
for encoder, weight in checkpoint_baseline['state_dict'].items():
    if 'backbone' in encoder:
        encoder = encoder.replace("backbone.","")
        encoder_baseline[encoder] = weight


#HRDA model
HRDA_checkpoint_path = '/home/yliang/work/DAFormer/pretrained/HRDA_GTA.pth'

HRDA_gta= torch.load(HRDA_checkpoint_path, map_location=torch.device(device))

encoder_HRDA_gta={}
for encoder, weight in HRDA_gta['state_dict'].items():
    if 'model.backbone' in encoder and 'ema_' not in encoder and 'imnet_' not in encoder:
        # print(encoder)
        encoder = encoder.replace("model.backbone.","") 
        weight = weight.float()
        encoder_HRDA_gta[encoder] = weight



selected_backbone = 'mit_b5' # import model and set weight

# 1.1 import model and set weight
model1 = BACKBONES.get(selected_backbone)(init_cfg = dict(type='Pretrained'))
model1.load_state_dict(encoder_baseline)
model1.eval()

HRDA_gta = BACKBONES.get(selected_backbone)(init_cfg = dict(type='Pretrained'))
HRDA_gta.load_state_dict(encoder_HRDA_gta)
HRDA_gta.eval()


# 2. Create the DataLoader
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(0)
np.random.seed(0)
random.seed(0)

# torch.manual_seed(42) # Set a random seed for reproducibility

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
num_images = 256
limited_dataset = itertools.islice(cityscapes_dataset, num_images)
limited_list = list(limited_dataset)

batch_size = 16
dataloader = DataLoader(limited_list,
                        batch_size=batch_size, 
                        shuffle=False,
                        worker_init_fn=seed_worker,
                        generator=g)


test_layer = []
for name, module in model1.named_modules():
    if 'mlp.drop' in name:
    # if 'attn.norm' in name:
        test_layer.append(name)
        print(name)

# test_layer = ['block3.39.mlp.dwconv.dwconv','block3.39.mlp.act','block3.39.mlp.fc2','block3.39.mlp.drop','norm3','norm4']
# test_layer = ['norm2','norm3','norm4']

cka = CKA(model1, HRDA_gta,
          model1_name="Baseline",   # good idea to provide names to avoid confusion
          model2_name="HRDA",   
          model1_layers=test_layer, # List of layers to extract features from
          model2_layers=test_layer, # extracts all layer features by default
          device=device)
        # device=device)

cka.compare(dataloader1=dataloader, dataloader2=dataloader) # secondary dataloader is optional

results = cka.export()  # returns a dict that contains model names, layer names
                        # and the CKA matrix

cka.plot_results(title=f"Mix-FFN",save_path="/home/yliang/work/DAFormer/save_file/similarity_comparision/cka_result/3_mix-ffn_blocks/BL_vs_HRDA_mlp_drop_n256_gpu.png")
