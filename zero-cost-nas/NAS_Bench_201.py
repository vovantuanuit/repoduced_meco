import argparse
import os

import time

from foresight.dataset import *
from foresight.models import nasbench2
from foresight.pruners import predictive
from foresight.weight_initializers import init_net
from foresight.pruners.measures.meco import *#get_score, get_score_grad, get_score_zico,get_score_gradfeature,get_score_grad_split,get_score_8x8
from xautodl.models import get_cell_based_tiny_net
import pickle
from tqdm import tqdm

import json
from scipy import stats

def get_num_classes(args):
    return 100 if args.dataset == 'cifar100' else 10 if args.dataset == 'cifar10' else 120


def parse_arguments():
    parser = argparse.ArgumentParser(description='Zero-cost Metrics for NAS-Bench-201')
    parser.add_argument('--api_loc', default='../data/NAS-Bench-201-v1_0-e61699.pth',
                        type=str, help='path to API')
    parser.add_argument('--outdir', default='./',
                        type=str, help='output directory')
    parser.add_argument('--search_space', default='tss',
                        type=str, help='output directory')
    parser.add_argument('--init_w_type', type=str, default='none',
                        help='weight initialization (before pruning) type [none, xavier, kaiming, zero, one]')
    parser.add_argument('--init_b_type', type=str, default='none',
                        help='bias initialization (before pruning) type [none, xavier, kaiming, zero, one]')
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--dataset', type=str, default='ImageNet16',
                        help='dataset to use [cifar10, cifar100, ImageNet16-120]')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index to work on')
    parser.add_argument('--data_size', type=int, default=32, help='data_size')
    parser.add_argument('--num_data_workers', type=int, default=2, help='number of workers for dataloaders')
    parser.add_argument('--dataload', type=str, default='appoint', help='random, grasp, appoint supported')
    parser.add_argument('--dataload_info', type=int, default=1,
                        help='number of batches to use for random dataload or number of samples per class for grasp dataload')
    parser.add_argument('--seed', type=int, default=84, help='pytorch manual seed')
    parser.add_argument('--write_freq', type=int, default=1, help='frequency of write to file')
    parser.add_argument('--start', type=int, default=0, help='start index')
    parser.add_argument('--end', type=int, default=0, help='end index')
    parser.add_argument('--noacc', default=False, action='store_true',
                        help='avoid loading NASBench2 api an instead load a pickle file with tuple (index, arch_str)')
    args = parser.parse_args()
    args.device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")
    return args


if __name__ == '__main__':
    args = parse_arguments()
    # print(args.device)
    path_save_json = "/home/tuanvovan/MeCo/zero-cost-nas/total_infor_only_meco_8x8_fea_cifar100_2.json"
    if args.noacc:
        api = pickle.load(open(args.api_loc,'rb'))
    else:
        from nas_201_api import NASBench201API as API
        api = API(args.api_loc)

    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    train_loader, val_loader = get_cifar_dataloaders(args.batch_size, args.batch_size, args.dataset, args.num_data_workers, resize=args.data_size)
    x, y = next(iter(train_loader))
    x = x.to(args.device)
    # x.requires_grad = True
    y = y.to(args.device)
    # random data
    # x = torch.rand((args.batch_size, 3, args.data_size, args.data_size))
    # y = 0

    cached_res = []
    test_acc = []
    meco_scores = []
    pre = 'cf' if 'cifar' in args.dataset else 'im'
    pfn = f'nb2_{args.search_space}_{pre}{get_num_classes(args)}_seed{args.seed}_dl{args.dataload}_dlinfo{args.dataload_info}_initw{args.init_w_type}_initb{args.init_b_type}_{args.batch_size}.p'
    op = os.path.join(args.outdir, pfn)

    end = len(api) if args.end == 0 else args.end
    # end = 100
    # loop over nasbench2 archs
    list_8x8_mecos =[]
    list_meco_scores = {}
    for i, arch_str in tqdm(enumerate(api)):
        if i < args.start:
            continue
        if i >= end:
            break

        res = {'i': i, 'arch': arch_str}
        # print(arch_str)
        if args.search_space == 'tss':
            net = nasbench2.get_model_from_arch_str(arch_str, get_num_classes(args))
            arch_str2 = nasbench2.get_arch_str_from_model(net)
            if arch_str != arch_str2:
                # print(arch_str)
                # print(arch_str2)
                raise ValueError
        elif args.search_space == 'sss':
            config = api.get_net_config(i, args.dataset)
            # print(config)
            net = get_cell_based_tiny_net(config)
        net.to(args.device)
        # print(net)
        

        init_net(net, args.init_w_type, args.init_b_type)
        net.train()
        # print(x.size(), y)
        measures, meco_layers,eigen_layers, layer_shape_C = get_score_Meco_8x8_opt_result(net, x, y, args.device,1)
        # list_8x8_mecos.append(result_list)

        res['meco'] = measures
        # list_meco_scores[i]=measures

        if not args.noacc:
            info = api.get_more_info(i, 'cifar10-valid' if args.dataset == 'cifar10' else args.dataset, iepoch=None,
                                     hp='200', is_random=False)
            # print(args.dataset)
            trainacc = info['train-accuracy']
            valacc = info['valid-accuracy']
            testacc = info['test-accuracy']

            res['trainacc'] = trainacc
            res['valacc'] = valacc
            res['testacc'] = testacc
            test_acc.append(testacc)
            meco_scores.append(measures)

        # print(res)
        cached_res.append(res)

        # write to file
        if i % args.write_freq == 0 or i == len(api) - 1 or i == 10:
            # print(f'writing {len(cached_res)} results to {op}')
            pf = open(op, 'ab')
            for cr in cached_res:
                pickle.dump(cr, pf)
            pf.close()
            cached_res = []

    # correlations = compute_scores(ytest=test_acc, test_pred=meco_scores)
    # kendalltau_corr = correlations['kendalltau']
    # spearman_corr = correlations['spearman']
    # pearson_corr = correlations['pearson']
        infor_mecos = {}
        trainacc = info['train-accuracy']
        valacc = info['valid-accuracy']
        testacc = info['test-accuracy']
        infor_mecos['meco'] = measures
        infor_mecos['moce_layers'] = meco_layers
        infor_mecos['eigen_list_layer'] = eigen_layers
        infor_mecos['layer_shapes'] = layer_shape_C
        infor_mecos['train-accuracy'] = trainacc
        infor_mecos['valid-accuracy']=valacc
        infor_mecos['test-accuracy']=testacc
        
        list_meco_scores[i]=infor_mecos
    spearman_corr = stats.spearmanr(meco_scores,test_acc)

    print('*'*50)
    # print('Validation accuracies: ', val_accs)
    print()
    # print('Zero Cost predictor scores: ', zc_scores)
    print('*'*50)
    print('Correlations between validation accuracies (ground truth) and Zero Cost predictor scores (prediction): ')
    # print('Kendall Tau correlation:', kendalltau_corr)
    print('Spearman correlation:', spearman_corr)
    # print('Pearson correlation:', pearson_corr)
    # with open("/home/tuanvovan/MeCo/zero-cost-nas/list_8_8_mecovalue_image16-120.pkl", "wb") as file:  # Open file in binary write mode
    #     pickle.dump(list_8x8_mecos, file)
    with open(path_save_json, "w") as file:  # Open file in binary write mode
        json.dump(list_meco_scores, file)