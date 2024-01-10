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

ImagenetDataProvider.DEFAULT_PATH = path_data_imagenet
run_config = ImagenetRunConfig(test_batch_size=512, n_worker=8)

""" Randomly sample a sub-network, 
    you can also manually set the sub-network using: 
        ofa_network.set_active_subnet(ks=7, e=6, d=4) 
"""

# print("Results: loss=%.5f,\t top1=%.1f,\t top5=%.1f" % (loss, top1, top5))
train_data, test_data, xshape, class_num = datasets.get_datasets('imagenet-1k', path_data_imagenet, 0)
print('num-calss ---:',class_num)
# valloader = torch.utils.data.DataLoader(
#         test_data, batch_size=1024, shuffle=False, num_workers=4)

trainloader = torch.utils.data.DataLoader(
        train_data, batch_size=64, shuffle=True, num_workers=4)
x, y = next(iter(trainloader))
device = torch.device("cuda:" + str(0) if torch.cuda.is_available() else "cpu")
x = x.to(device)
# x.requires_grad = True
y = y.to(device)
list_meco_scores = {}
meco_scores = []
test_acc=[]
for i in range(1000):
    st = time.time()
    print(i)
    ofa_network.sample_active_subnet()
    subnet = ofa_network.get_active_subnet(preserve_weight=True)

    """ Test sampled subnet 
    """
    run_manager = RunManager(".tmp/eval_subnet", subnet, run_config, init=False)
    # assign image size: 128, 132, ..., 224
    run_config.data_provider.assign_active_img_size(224)
    run_manager.reset_running_statistics(net=subnet)

    subnet.train()
    subnet.to(device)
    measures, meco_layers,eigen_layers,layer_shape_C = get_score_Meco_result(subnet, x, y, device,1)
    print('meco: ',measures)
    with torch.no_grad():
        subnet.eval()
        loss, (top1, top5) = run_manager.validate(net=subnet)
    print('top-1 Acc: ',top1)
    del subnet
    infor_mecos = {}
        
    infor_mecos['meco'] = measures
    infor_mecos['moce_layers'] = meco_layers
    infor_mecos['eigen_list_layer'] = eigen_layers
    infor_mecos['layers_shape'] = layer_shape_C
    infor_mecos['Test Accuracy'] = top1
    list_meco_scores[i] = infor_mecos
    meco_scores.append(measures)
    test_acc.append(top1)

    end = time.time()
    print('time: ',end-st)
with open(path_save_json, "w") as file:  # Open file in binary write mode
    json.dump(list_meco_scores, file)
        
spearman_corr = stats.spearmanr(meco_scores,test_acc)

print('*'*50)
# print('Validation accuracies: ', val_accs)
print()
# print('Zero Cost predictor scores: ', zc_scores)
print('*'*50)
print('Correlations between validation accuracies (ground truth) and Zero Cost predictor scores (prediction): ')
# print('Kendall Tau correlation:', kendalltau_corr)
print('Spearman correlation:', spearman_corr)