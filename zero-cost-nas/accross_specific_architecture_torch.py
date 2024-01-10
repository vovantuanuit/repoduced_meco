import torch
from foresight.pruners.measures.meco import *
from ZiCo.xautodl import datasets
import json
from foresight.weight_initializers import init_net
from torchvision.models import mobilenet_v2,resnet18,efficientnet_b0,maxvit_t
from timm import create_model


# Create a new model with desired number of classes (replace 1000 with your number)




path_save_json = '/home/tuanvovan/MeCo/zero-cost-nas/total_infor_meco_maxVIT_torch_image1k_pretraied_False.json'

train_data, test_data, xshape, class_num = datasets.get_datasets('imagenet-1k', '/home/tuanvovan/MeCo/zero-cost-nas/ZiCo/dataset/imagenet', 0)

trainloader = torch.utils.data.DataLoader(
        train_data, batch_size=8, shuffle=True, num_workers=4)
x, y = next(iter(trainloader))
device = torch.device("cuda:" + str(0) if torch.cuda.is_available() else "cpu")
x = x.to(device)
# x.requires_grad = True
y = y.to(device)
list_meco_scores = {}
# model = maxvit_t(num_classes=1000,pretrained=True)
model = create_model('maxvit_base_tf_224.in1k', pretrained=False)
model.train()
model.to(device)
for i in range(1):
    print(i)
    measures, meco_layers,eigen_layers,layer_shape_C = get_score_Meco_result(model, x, y, device,1)
    infor_mecos = {}
        
    infor_mecos['meco'] = measures
    infor_mecos['moce_layers'] = meco_layers
    infor_mecos['eigen_list_layer'] = eigen_layers
    infor_mecos['layer_shape_C'] = layer_shape_C
    list_meco_scores[i] = infor_mecos

with open(path_save_json, "w") as file:  # Open file in binary write mode
    json.dump(list_meco_scores, file)
        
