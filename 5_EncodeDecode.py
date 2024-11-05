import torch
from mmseg.models.builder import SEGMENTORS

if torch.cuda.is_available():
    device = torch.device('cuda')
    print('CUDA is available. Using GPU...')
else:
    device = torch.device('cpu')
    print('CUDA is not available. Using CPU...')
    


'''
# 1. baseline---------------------------------------------------------------
BL_decode_encode = SEGMENTORS.get('EncoderDecoder')(
    backbone = dict(type='mit_b5',init_cfg=dict(type='Pretrained')),
    decode_head= dict(type='SegFormerHead' ,init_cfg=dict(type='Pretrained'),
                          in_channels=[64, 128, 320, 512],  channels=128,  num_classes=19,
    in_index=[1, 2, 3, 4],
    decoder_params=dict(
            embed_dim=768,
            conv_kernel_size=1
        ),
    norm_cfg=dict(
            type='SyncBN',
            requires_grad=True
        )
))

baseline_checkpoint_path = '/home/yliang/work/DAFormer/pretrained/BaselineA.pth'
checkpoint = torch.load(baseline_checkpoint_path, map_location=device)


state_dict = checkpoint['state_dict']
# torch.save(state_dict,'/home/yliang/work/DAFormer/model_info/BL_state_dict.pt')
state_dict = torch.load('/home/yliang/work/DAFormer/model_info/BL_state_dict.pt')

BL_state_dict = {}

for key, weight in state_dict.items():
    if 'linear_c1' in key:
        key = key.replace("linear_c1", "linear_c.1") 
        weight = weight.float()
        BL_state_dict[key] = weight
    elif 'linear_c2' in key:
        key = key.replace("linear_c2", "linear_c.2") 
        weight = weight.float()
        BL_state_dict[key] = weight
    elif 'linear_c3' in key:
        key = key.replace("linear_c3", "linear_c.3") 
        weight = weight.float()
        BL_state_dict[key] = weight
    elif 'linear_c4' in key:
        key = key.replace("linear_c4", "linear_c.4") 
        weight = weight.float()
        BL_state_dict[key] = weight
    else:
        weight = weight.float()
        BL_state_dict[key] = weight

BL_decode_encode.load_state_dict(BL_state_dict)
bl_decode_encode=BL_decode_encode.to(device)
bl_decode_encode.eval()



# explore---------------------------------------

from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot

# clean layer's name
# checkpoint_path = '/home/yliang/work/DAFormer/pretrained/BaselineA.pth'
# dict=torch.load(checkpoint_path, device)
# dict['state_dict'] = BL_state_dict
# torch.save(dict, '/home/yliang/work/DAFormer/pretrained/BaselineA_modified.pth')

config_path = '/home/yliang/work/DAFormer/configs/_base_/models/segformer_b5.py'
checkpoint_path = '/home/yliang/work/DAFormer/pretrained/BaselineA_modified.pth'
segformer_b5 = init_segmentor(config_path, checkpoint_path, device=device)

input_data = torch.randn(1, 3, 256, 512)
result = segformer_b5.forward_dummy(input_data)



img_path = '/home/yliang/work/DAFormer/demo/demo.png'


result = inference_segmentor(segformer_b5, img_path)

vis_image = show_result_pyplot(segformer_b5, img_path, result)

vis_iamge = show_result_pyplot(segformer_b5, img_path, result, out_file='/home/yliang/work/DAFormer/save_file/encode_decode_feature/baseline/result_test.png')



input_data = torch.randn(1, 3, 256, 512)
img_metas = [{} for _ in range(1)]
gt_semantic_seg = [torch.zeros_like(input_data[0][0])] * 1
segformer_b5(input_data, img_metas=img_metas, gt_semantic_seg=gt_semantic_seg)



#-----------------------------------------------





# 2.HRDA-----------------------------------------------------------------
hrda_decode_encode = SEGMENTORS.get('EncoderDecoder')(
    backbone = dict(type='mit_b5',init_cfg=dict(type='Pretrained')),
    decode_head= dict(type='DAFormerHead' ,init_cfg=dict(type='Pretrained'),
                          in_channels=[64, 128, 320, 512],  channels=256,  
                          num_classes=19, in_index=[0, 1, 2, 3],
                        decoder_params=dict(
                        embed_dims=256, embed_cfg=dict(type='mlp', act_cfg=None, norm_cfg=None),
                        embed_neck_cfg=dict(type='mlp', act_cfg=None, norm_cfg=None),
                        fusion_cfg=dict(type='aspp',sep=True, dilations=(1, 6, 12, 18),
                                        pool=False, act_cfg=dict(type='ReLU'),
                                        norm_cfg=dict(type='BN', requires_grad=True)
                                )),
                            norm_cfg=dict(
                                    type='BN',
                                    requires_grad=True
                                )
                        ))


hrda_gta_checkpoint_path = '/home/yliang/work/DAFormer/pretrained/HRDA_GTA.pth'
checkpoint = torch.load(hrda_gta_checkpoint_path, map_location=device)

filtered_dict = checkpoint['state_dict']

filtered_dict = {key: value for key, value in filtered_dict.items() if 'imnet' not in key and 'ema' not in key}

HRDA_state_dict={}
for key, weight in filtered_dict.items():
    if 'model.backbone' in key:
        key = key.replace("model.backbone", "backbone") 
        weight = weight.float()
        HRDA_state_dict[key] = weight
    if 'model.decode_head.head' in key:
        key = key.replace("model.decode_head.head", "decode_head") 
        weight = weight.float()
        HRDA_state_dict[key] = weight


# torch.save(HRDA_state_dict,'/home/yliang/work/DAFormer/model_info/HRDA_state_dict.pt')
HRDA_state_dict = torch.load('/home/yliang/work/DAFormer/model_info/HRDA_state_dict.pt')

hrda_decode_encode.load_state_dict(HRDA_state_dict)
hrda_gta5_decode_encode=hrda_decode_encode.to(device)
hrda_gta5_decode_encode.eval()
'''

#hrda_synthia
hrda_synthia_checkpoint_path = '/home/yliang/work/DAFormer/pretrained/HRDA_Synthia.pth'
checkpoint = torch.load(hrda_synthia_checkpoint_path, map_location=device)

filtered_dict = checkpoint['state_dict']

filtered_dict = {key: value for key, value in filtered_dict.items() if 'imnet' not in key and 'ema' not in key}

HRDA_Synthia_state_dict={}
for key, weight in filtered_dict.items():
    if 'model.backbone' in key:
        key = key.replace("model.backbone", "backbone") 
        weight = weight.float()
        HRDA_Synthia_state_dict[key] = weight
    if 'model.decode_head.head' in key:
        key = key.replace("model.decode_head.head", "decode_head") 
        weight = weight.float()
        HRDA_Synthia_state_dict[key] = weight


torch.save(HRDA_Synthia_state_dict,'/home/yliang/work/DAFormer/model_info/HRDA_Synthia_state_dict.pt')
# HRDA_state_dict = torch.load('/home/yliang/work/DAFormer/model_info/HRDA_Synthia_state_dict.pt')
