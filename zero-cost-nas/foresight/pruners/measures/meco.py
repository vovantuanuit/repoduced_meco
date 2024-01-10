# Copyright 2021 Samsung Electronics Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
import copy
import time

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

import numpy as np
import torch
from torch import nn
lossfunc = nn.CrossEntropyLoss().cuda()

from . import measure

import pickle

def get_score_zico(net, x, target, device, split_data):
    result_list = []
    
    outputs = net.forward(x)
    loss = lossfunc(outputs, target)
    loss.backward()
    def forward_hook(module, data_input, data_output):
        # fea_or = data_output[0]
        fea = data_output[0].detach()
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            # grad = fea_or.grad
            
            grad = module.weight.grad.data.cpu().reshape( -1).numpy()
            nsr_std = np.std(grad, axis=0)
            # print(nsr_std)
            # nonzero_idx = np.nonzero(nsr_std)[0]
            nsr_mean_abs = np.mean(np.abs(grad), axis=0)
            tmpsum = np.sum(nsr_mean_abs/nsr_std)
            zico = np.log(tmpsum)
        else:
            zico = 1.0
        
        # fea = fea*grad
        fea = fea.reshape(fea.shape[0], -1)
        
        corr = torch.corrcoef(fea)
        corr[torch.isnan(corr)] = 0
        corr[torch.isinf(corr)] = 0
        values = torch.linalg.eig(corr)[0]
        # result = np.real(np.min(values)) / np.real(np.max(values))
        result = torch.min(torch.real(values))*zico
        result_list.append(result)

    for name, modules in net.named_modules():
        # print(modules)
        # print('----------------------------------------------------------')
        modules.register_forward_hook(forward_hook)



    N = x.shape[0]
    for sp in range(split_data):
        st = sp * N // split_data
        en = (sp + 1) * N // split_data
        y = net(x[st:en])
    results = torch.tensor(result_list)
    results = results[torch.logical_not(torch.isnan(results))]
    v = torch.sum(results)
    result_list.clear()
    return v.item()


def get_score_Meco(net, x, target, device, split_data):
    result_list = []
    
    def forward_hook(module, data_input, data_output):
        # print('---day')
        fea = data_output[0].detach()
        fea = fea.reshape(fea.shape[0], -1)
        corr = torch.corrcoef(fea)
        corr[torch.isnan(corr)] = 0
        corr[torch.isinf(corr)] = 0
        values = torch.linalg.eig(corr)[0]
        # result = np.real(np.min(values)) / np.real(np.max(values))
        result = torch.min(torch.real(values))
        result_list.append(result)

    for name, modules in net.named_modules():
        modules.register_forward_hook(forward_hook)



    N = x.shape[0]
    for sp in range(split_data):
        st = sp * N // split_data
        en = (sp + 1) * N // split_data
        y = net(x[st:en])
    results = torch.tensor(result_list)
    results = results[torch.logical_not(torch.isnan(results))]
    v = torch.sum(results)
    result_list.clear()
    return v.item()

def get_score_Meco_result(net, x, target, device, split_data):
    result_list = []
    meco_layers = []
    eigen_layers = []
    layer_shape_C = []
    def forward_hook(module, data_input, data_output):
        # print('---day')
        fea = data_output[0].detach()
        fea = fea.reshape(fea.shape[0], -1)
        corr = torch.corrcoef(fea)
        corr[torch.isnan(corr)] = 0
        corr[torch.isinf(corr)] = 0
        values = torch.linalg.eig(corr)[0]
        # result = np.real(np.min(values)) / np.real(np.max(values))
        result = torch.min(torch.real(values))
        result_list.append(result)
        meco_layers.append(result.item())
        eigen_layers.append(torch.real(values).tolist())
        layer_shape_C.append((fea.shape[0],fea.shape[0]))

    for name, modules in net.named_modules():
        modules.register_forward_hook(forward_hook)



    N = x.shape[0]
    for sp in range(split_data):
        st = sp * N // split_data
        en = (sp + 1) * N // split_data
        y = net(x[st:en])
    results = torch.tensor(result_list)
    results = results[torch.logical_not(torch.isnan(results))]
    v = torch.sum(results)
    result_list.clear()
    return v.item(),meco_layers,eigen_layers,layer_shape_C


def get_score_Meco_result_heatmap(net, x, target, device, split_data):
    result_list = []
    meco_layers = []
    heamaps = []
    def forward_hook(module, data_input, data_output):
        # print('---day')
        fea = data_output[0].detach()
        fea = fea.reshape(fea.shape[0], -1)
        
        corr = torch.corrcoef(fea)
        corr[torch.isnan(corr)] = 0
        corr[torch.isinf(corr)] = 0
        values = torch.linalg.eig(corr)[0]
        # result = np.real(np.min(values)) / np.real(np.max(values))
        result = torch.min(torch.real(values))
        result_list.append(result)
        meco_layers.append(result.item())
        heamaps.append(corr)

    for name, modules in net.named_modules():
        modules.register_forward_hook(forward_hook)



    N = x.shape[0]
    for sp in range(split_data):
        st = sp * N // split_data
        en = (sp + 1) * N // split_data
        y = net(x[st:en])
    results = torch.tensor(result_list)
    results = results[torch.logical_not(torch.isnan(results))]
    v = torch.sum(results)
    result_list.clear()
    return v.item(),meco_layers,heamaps

def get_score_Meco_input_random_result(net, x, target, device, split_data):
    result_list = []
    meco_layers = []
    eigen_layers = []
    def forward_hook(module, data_input, data_output):
        # print('---day')
        fea = data_output[0].detach()
        fea = fea.reshape(fea.shape[0], -1)
        corr = torch.corrcoef(fea)
        corr[torch.isnan(corr)] = 0
        corr[torch.isinf(corr)] = 0
        values = torch.linalg.eig(corr)[0]
        # result = np.real(np.min(values)) / np.real(np.max(values))
        result = torch.min(torch.real(values))
        result_list.append(result)
        meco_layers.append(result.item())
        eigen_layers.append(torch.real(values).tolist())

    for name, modules in net.named_modules():
        modules.register_forward_hook(forward_hook)


    

# Generate a tensor with random values from a standard Gaussian distribution
    x = torch.randn(size=x.shape).to('cuda')
    N = x.shape[0]
    for sp in range(split_data):
        st = sp * N // split_data
        en = (sp + 1) * N // split_data
        y = net(x[st:en])
    results = torch.tensor(result_list)
    results = results[torch.logical_not(torch.isnan(results))]
    v = torch.sum(results)
    result_list.clear()
    return v.item(),meco_layers,eigen_layers

def get_score_Meo_8x8_opt(net, x, target, device, split_data):
    result_list = []
    temp_list = []
    def forward_hook(module, data_input, data_output):
        # print('---day')
        fea = data_output[0].detach()
        fea = fea.reshape(fea.shape[0], -1)
        random_indices_8_a = torch.randperm(fea.shape[0])[:8]  # Get 8 random indices
        random_tensor_8_a_fea = fea[random_indices_8_a]
        corr = torch.corrcoef(random_tensor_8_a_fea)
        # print(corr.shape)
        corr[torch.isnan(corr)] = 0
        corr[torch.isinf(corr)] = 0
        values = torch.linalg.eig(corr)[0]
        # result = np.real(np.min(values)) / np.real(np.max(values))
        result = torch.min(torch.real(values))
        result_list.append(result)
        temp_list.append(result.item())

    for name, modules in net.named_modules():
        modules.register_forward_hook(forward_hook)



    N = x.shape[0]
    for sp in range(split_data):
        st = sp * N // split_data
        en = (sp + 1) * N // split_data
        y = net(x[st:en])
    
    results = torch.tensor(result_list)
    results = results[torch.logical_not(torch.isnan(results))]
    v = torch.sum(results)
    # result_list.clear()
    return v.item(),temp_list

def get_score_Meco_8x8_opt_weight_result(net, x, target, device, split_data):
    result_list = []
    meco_layers = []
    eigen_layers = []
    layer_shape_C = []
    def forward_hook(module, data_input, data_output):
        # print('---day')
        fea = data_output[0].detach()
        fea = fea.reshape(fea.shape[0], -1)
        random_indices_8_a = torch.randperm(fea.shape[0])[:8]  # Get 8 random indices
        random_tensor_8_a_fea = fea[random_indices_8_a]
        corr = torch.corrcoef(random_tensor_8_a_fea)
        # print(corr.shape)
        corr[torch.isnan(corr)] = 0
        corr[torch.isinf(corr)] = 0
        values = torch.linalg.eig(corr)[0]
        # result = np.real(np.min(values)) / np.real(np.max(values))
        result = (fea.shape[0]/8)*torch.min(torch.real(values))
        result_list.append(result)
        meco_layers.append(result.item())
        eigen_layers.append(torch.real(values).tolist())
        layer_shape_C.append((fea.shape[0],fea.shape[0]))
        # print(layer_shape_C)

    for name, modules in net.named_modules():
        modules.register_forward_hook(forward_hook)



    N = x.shape[0]
    for sp in range(split_data):
        st = sp * N // split_data
        en = (sp + 1) * N // split_data
        y = net(x[st:en])
    
    results = torch.tensor(result_list)
    results = results[torch.logical_not(torch.isnan(results))]
    v = torch.sum(results)
    result_list.clear()
    return v.item(),meco_layers,eigen_layers,layer_shape_C

def get_score_Meco_8x8_opt_result(net, x, target, device, split_data):
    result_list = []
    meco_layers = []
    eigen_layers = []
    layer_shape_C = []
    def forward_hook(module, data_input, data_output):
        # print('---day')
        fea = data_output[0].detach()
        fea = fea.reshape(fea.shape[0], -1)
        random_indices_8_a = torch.randperm(fea.shape[0])[:8]  # Get 8 random indices
        random_tensor_8_a_fea = fea[random_indices_8_a]
        corr = torch.corrcoef(random_tensor_8_a_fea)
        # print(corr.shape)
        corr[torch.isnan(corr)] = 0
        corr[torch.isinf(corr)] = 0
        values = torch.linalg.eig(corr)[0]
        # result = np.real(np.min(values)) / np.real(np.max(values))
        result = torch.min(torch.real(values))
        result_list.append(result)
        meco_layers.append(result.item())
        eigen_layers.append(torch.real(values).tolist())
        layer_shape_C.append((fea.shape[0],fea.shape[0]))
        # print(layer_shape_C)

    for name, modules in net.named_modules():
        modules.register_forward_hook(forward_hook)



    N = x.shape[0]
    for sp in range(split_data):
        st = sp * N // split_data
        en = (sp + 1) * N // split_data
        y = net(x[st:en])
    
    results = torch.tensor(result_list)
    results = results[torch.logical_not(torch.isnan(results))]
    v = torch.sum(results)
    result_list.clear()
    return v.item(),meco_layers,eigen_layers,layer_shape_C



def get_score_Meco_grad(net, x, target, device, split_data):
    result_list = []
    for param in net.parameters():
        param.requires_grad = True
    def forward_hook(module, data_input, data_output):
        dydx = torch.autograd.grad(outputs=data_output, inputs=data_input,
            grad_outputs=torch.ones_like(data_output),
            retain_graph=True)[0]

        fea = data_output[0].detach()
        grad = dydx[0].detach()
        # print(grad)
        if fea.shape == grad.shape:
            # print(fea.shape)
            # print(grad.shape)
            # print('vo day nhe')
            # print('orginal:',fea)
            # mix_grad_fea = torch.cat([fea, grad], dim=1)
            mix_grad_fea = torch.cat([fea, grad], dim=0)
            # mix_grad_fea = fea*grad
            # print('mix', mix_grad_fea.shape)
        else:
            mix_grad_fea = fea
        fea_mix = mix_grad_fea.reshape(mix_grad_fea.shape[0], -1)
        corr = torch.corrcoef(fea_mix)
        corr[torch.isnan(corr)] = 0
        corr[torch.isinf(corr)] = 0
        values = torch.linalg.eig(corr)[0]
        # result = np.real(np.min(values)) / np.real(np.max(values))
        result = torch.min(torch.real(values))
        result_list.append(result)
  

    for name, modules in net.named_modules():
        modules.register_forward_hook(forward_hook)

    N = x.shape[0]
    for sp in range(split_data):
        st = sp * N // split_data
        en = (sp + 1) * N // split_data
        
        in_  = x[st:en]
        # print(in_)
        in_ = torch.tensor(in_, requires_grad=True)
        y = net(in_)
        loss = lossfunc(y, target[st:en])
        loss.backward()
    results = torch.tensor(result_list)
    results = results[torch.logical_not(torch.isnan(results))]
    v = torch.sum(results)
    result_list.clear()
    return v.item()

def get_score_Meco_grad_result(net, x, target, device, split_data):
    result_list = []
    meco_layers = []
    eigen_layers = []
    for param in net.parameters():
        param.requires_grad = True
    def forward_hook(module, data_input, data_output):
        dydx = torch.autograd.grad(outputs=data_output, inputs=data_input,
            grad_outputs=torch.ones_like(data_output),
            retain_graph=True)[0]

        fea = data_output[0].detach()
        grad = dydx[0].detach()
        # print(grad)
        if fea.shape == grad.shape:
            # print(fea.shape)
            # print(grad.shape)
            # print('vo day nhe')
            # print('orginal:',fea)
            mix_grad_fea = torch.cat([fea, grad], dim=0)
            # mix_grad_fea = fea*grad
            # print('mix', grad)
        else:
            mix_grad_fea = fea
        fea_mix = mix_grad_fea.reshape(mix_grad_fea.shape[0], -1)
        corr = torch.corrcoef(fea_mix)
        corr[torch.isnan(corr)] = 0
        corr[torch.isinf(corr)] = 0
        values = torch.linalg.eig(corr)[0]
        # result = np.real(np.min(values)) / np.real(np.max(values))
        result = torch.min(torch.real(values))
        result_list.append(result)
        meco_layers.append(result.item())
        eigen_layers.append(torch.real(values).tolist())
  

    for name, modules in net.named_modules():
        modules.register_forward_hook(forward_hook)

    N = x.shape[0]
    for sp in range(split_data):
        st = sp * N // split_data
        en = (sp + 1) * N // split_data
        
        in_  = x[st:en]
        # print(in_)
        in_ = torch.tensor(in_, requires_grad=True)
        y = net(in_)
        loss = lossfunc(y, target[st:en])
        loss.backward()
    results = torch.tensor(result_list)
    results = results[torch.logical_not(torch.isnan(results))]
    v = torch.sum(results)
    result_list.clear()
    return v.item(),meco_layers,eigen_layers


def get_score_Meco_grad_input_random_result(net, x, target, device, split_data):
    result_list = []
    meco_layers = []
    eigen_layers = []
    for param in net.parameters():
        param.requires_grad = True
    def forward_hook(module, data_input, data_output):
        dydx = torch.autograd.grad(outputs=data_output, inputs=data_input,
            grad_outputs=torch.ones_like(data_output),
            retain_graph=True)[0]

        fea = data_output[0].detach()
        grad = dydx[0].detach()
        # print(grad)
        if fea.shape == grad.shape:
            # print(fea.shape)
            # print(grad.shape)
            # print('vo day nhe')
            # print('orginal:',fea)
            mix_grad_fea = torch.cat([fea, grad], dim=0)
            # mix_grad_fea = fea*grad
            # print('mix', grad)
        else:
            mix_grad_fea = fea
        fea_mix = mix_grad_fea.reshape(mix_grad_fea.shape[0], -1)
        corr = torch.corrcoef(fea_mix)
        corr[torch.isnan(corr)] = 0
        corr[torch.isinf(corr)] = 0
        values = torch.linalg.eig(corr)[0]
        # result = np.real(np.min(values)) / np.real(np.max(values))
        result = torch.min(torch.real(values))
        result_list.append(result)
        meco_layers.append(result.item())
        eigen_layers.append(torch.real(values).tolist())
  

    for name, modules in net.named_modules():
        modules.register_forward_hook(forward_hook)
    x=torch.randn(size=x.shape).to('cuda')
    N = x.shape[0]
    for sp in range(split_data):
        st = sp * N // split_data
        en = (sp + 1) * N // split_data
        
        in_  = x[st:en]
        # print(in_)
        in_ = torch.tensor(in_, requires_grad=True)
        y = net(in_)
        loss = lossfunc(y, target[st:en])
        loss.backward()
    results = torch.tensor(result_list)
    results = results[torch.logical_not(torch.isnan(results))]
    v = torch.sum(results)
    result_list.clear()
    return v.item(),meco_layers,eigen_layers




def get_score_gradoffeature(net, x, target, device, split_data):
    result_list = []
    for param in net.parameters():
        param.requires_grad = True
    def forward_hook(module, data_input, data_output):
        dydx = torch.autograd.grad(outputs=data_output, inputs=data_input,
            grad_outputs=torch.ones_like(data_output),
            retain_graph=True)[0]

        # fea = data_output[0].detach()
        grad = dydx[0].detach()
        # print(grad)
        # if fea.shape == grad.shape:
        #     # print(fea.shape)
        #     # print(grad.shape)
        #     # print('vo day nhe')
        #     # print('orginal:',fea)
        #     mix_grad_fea = torch.cat([fea, grad], dim=1)
        #     # mix_grad_fea = fea*grad
        #     # print('mix', grad)
        # else:
        #     mix_grad_fea = fea
        fea_grad = grad.reshape(grad.shape[0], -1)
        corr = torch.corrcoef(fea_grad)
        corr[torch.isnan(corr)] = 0
        corr[torch.isinf(corr)] = 0
        values = torch.linalg.eig(corr)[0]
        # result = np.real(np.min(values)) / np.real(np.max(values))
        result = torch.min(torch.real(values))
        result_list.append(result)
  

    for name, modules in net.named_modules():
        modules.register_forward_hook(forward_hook)

    N = x.shape[0]
    for sp in range(split_data):
        st = sp * N // split_data
        en = (sp + 1) * N // split_data
        
        in_  = x[st:en]
        # print(in_)
        in_ = torch.tensor(in_, requires_grad=True)
        y = net(in_)
        loss = lossfunc(y, target[st:en])
        loss.backward()
    results = torch.tensor(result_list)
    results = results[torch.logical_not(torch.isnan(results))]
    v = torch.sum(results)
    result_list.clear()
    return v.item()

def get_score_gradoffeature_result(net, x, target, device, split_data):
    result_list = []
    meco_layers = []
    eigen_layers = []
    for param in net.parameters():
        param.requires_grad = True
    def forward_hook(module, data_input, data_output):
        dydx = torch.autograd.grad(outputs=data_output, inputs=data_input,
            grad_outputs=torch.ones_like(data_output),
            retain_graph=True)[0]

        # fea = data_output[0].detach()
        grad = dydx[0].detach()
        # print(grad)
        # if fea.shape == grad.shape:
        #     # print(fea.shape)
        #     # print(grad.shape)
        #     # print('vo day nhe')
        #     # print('orginal:',fea)
        #     mix_grad_fea = torch.cat([fea, grad], dim=1)
        #     # mix_grad_fea = fea*grad
        #     # print('mix', grad)
        # else:
        #     mix_grad_fea = fea
        fea_grad = grad.reshape(grad.shape[0], -1)
        corr = torch.corrcoef(fea_grad)
        corr[torch.isnan(corr)] = 0
        corr[torch.isinf(corr)] = 0
        values = torch.linalg.eig(corr)[0]
        # result = np.real(np.min(values)) / np.real(np.max(values))
        result = torch.min(torch.real(values))
        result_list.append(result)
        meco_layers.append(result.item())
        eigen_layers.append(torch.real(values).tolist())
  

    for name, modules in net.named_modules():
        modules.register_forward_hook(forward_hook)

    N = x.shape[0]
    for sp in range(split_data):
        st = sp * N // split_data
        en = (sp + 1) * N // split_data
        
        in_  = x[st:en]
        # print(in_)
        in_ = torch.tensor(in_, requires_grad=True)
        y = net(in_)
        loss = lossfunc(y, target[st:en])
        loss.backward()
    results = torch.tensor(result_list)
    results = results[torch.logical_not(torch.isnan(results))]
    v = torch.sum(results)
    result_list.clear()
    return v.item(),meco_layers,eigen_layers



def get_score_gradoffeature_input_random_result(net, x, target, device, split_data):
    result_list = []
    meco_layers = []
    eigen_layers = []
    for param in net.parameters():
        param.requires_grad = True
    def forward_hook(module, data_input, data_output):
        dydx = torch.autograd.grad(outputs=data_output, inputs=data_input,
            grad_outputs=torch.ones_like(data_output),
            retain_graph=True)[0]

        # fea = data_output[0].detach()
        grad = dydx[0].detach()
        # print(grad)
        # if fea.shape == grad.shape:
        #     # print(fea.shape)
        #     # print(grad.shape)
        #     # print('vo day nhe')
        #     # print('orginal:',fea)
        #     mix_grad_fea = torch.cat([fea, grad], dim=1)
        #     # mix_grad_fea = fea*grad
        #     # print('mix', grad)
        # else:
        #     mix_grad_fea = fea
        fea_grad = grad.reshape(grad.shape[0], -1)
        corr = torch.corrcoef(fea_grad)
        corr[torch.isnan(corr)] = 0
        corr[torch.isinf(corr)] = 0
        values = torch.linalg.eig(corr)[0]
        # result = np.real(np.min(values)) / np.real(np.max(values))
        result = torch.min(torch.real(values))
        result_list.append(result)
        meco_layers.append(result.item())
        eigen_layers.append(torch.real(values).tolist())
  

    for name, modules in net.named_modules():
        modules.register_forward_hook(forward_hook)
    x = torch.randn(size=x.shape)
    N = x.shape[0]
    for sp in range(split_data):
        st = sp * N // split_data
        en = (sp + 1) * N // split_data
        
        in_  = x[st:en]
        # print(in_)
        in_ = torch.tensor(in_, requires_grad=True)
        y = net(in_)
        loss = lossfunc(y, target[st:en])
        loss.backward()
    results = torch.tensor(result_list)
    results = results[torch.logical_not(torch.isnan(results))]
    v = torch.sum(results)
    result_list.clear()
    return v.item(),meco_layers,eigen_layers

@measure('meco', bn=True)
def compute_meco(net, inputs, targets, split_data=1, loss_fn=None):
    device = inputs.device
    # Compute gradients (but don't apply them)
    net.zero_grad()

    try:
        meco = get_score(net, inputs, targets, device, split_data=split_data)
    except Exception as e:
        print(e)
        meco = np.nan, None
    return meco
