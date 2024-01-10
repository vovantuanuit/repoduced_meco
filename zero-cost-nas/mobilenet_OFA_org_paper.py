import torch
from foresight.pruners.measures.meco import *
from ZiCo.xautodl import datasets
import json
from foresight.weight_initializers import init_net
from ZiCo.ts_train_image_classification import *
import time
import os
import torch
import argparse

from once_for_all.ofa.imagenet_classification.data_providers.imagenet import ImagenetDataProvider
from once_for_all.ofa.imagenet_classification.run_manager import ImagenetRunConfig, RunManager
from once_for_all.ofa.model_zoo import ofa_net


path_data_imagenet = '/home/tuanvovan/MeCo/zero-cost-nas/ZiCo/dataset/imagenet'

path_save_json = "/home/tuanvovan/MeCo/zero-cost-nas/save_data.json"
ofa_network = ofa_net('ofa_mbv3_d234_e346_k357_w1.0', pretrained=True)


train_data, test_data, xshape, class_num = datasets.get_datasets('imagenet-1k', path_data_imagenet, 0)

trainloader = torch.utils.data.DataLoader(
        train_data, batch_size=64, shuffle=True, num_workers=4)
x, y = next(iter(trainloader))
device = torch.device("cuda:" + str(0) if torch.cuda.is_available() else "cpu")
x = x.to(device)
# x.requires_grad = True
y = y.to(device)
list_meco_scores = {}
for i in range(1000):
    print(i)
    ofa_network.sample_active_subnet()
    subnet = ofa_network.get_active_subnet(preserve_weight=True)
    subnet.train()
    subnet.to(device)
    measures, meco_layers,eigen_layers, layer_shape_C = get_score_Meco_result(subnet, x, y, device,1)
    infor_mecos = {}
        
    infor_mecos['meco'] = measures
    infor_mecos['moce_layers'] = meco_layers
    infor_mecos['eigen_list_layer'] = eigen_layers
    infor_mecos['layer_shape_C'] = layer_shape_C
    list_meco_scores[i] = infor_mecos

with open(path_save_json, "w") as file:  # Open file in binary write mode
    json.dump(list_meco_scores, file)
        
