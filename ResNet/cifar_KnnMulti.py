#! /usr/bin/env python3

import torch

import os

from util.data_loader import get_loader_in, get_loader_out
import models
import numpy as np
import torch.nn.functional as F
import time
import datetime
from util.args_loader import get_args
from util import metrics
import faiss
from util.logger import Logger
import json
import sys
from tqdm import tqdm
from sklearn.metrics import auc
import random
from sklearn.metrics import roc_auc_score
import math
import scipy.stats as ss
from sklearn.metrics import roc_auc_score,roc_curve
from util.model_loader import get_model



torch.manual_seed(1)
torch.cuda.manual_seed(1)
np.random.seed(1)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

args = get_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

loader_in_dict = get_loader_in(args, config_type="eval", split=('train', 'val'))
trainloaderIn, testloaderIn, num_classes = loader_in_dict.train_loader, loader_in_dict.val_loader, loader_in_dict.num_classes
model = get_model(args, num_classes, load_ckpt=True)
#model = getattr(models,args.model_arch)()  #修改部分******************************************************
#model = torch.nn.DataParallel(model).cuda()
#model.load_state_dict(torch.load(args.file),strict=False)
#********************************************************************************************

batch_size = args.batch_size

FORCE_RUN = True

if args.in_dataset=='imagenet':
    dummy_input = torch.zeros((1, 3, 224, 224)).cuda()
else:
    dummy_input = torch.zeros((1, 3, 32, 32)).cuda()
    
score, feature_list = model.feature_list(dummy_input)
featdims = [feat.shape[1] for feat in feature_list]

begin = time.time()

for split, in_loader in [('train', trainloaderIn), ('val', testloaderIn),]:

    cache_name = f"cache/{args.in_dataset}_{split}_{args.name}_in_alllayers.npy"
    if FORCE_RUN or not os.path.exists(cache_name):

        feat_log = np.zeros((len(in_loader.dataset), sum(featdims)))

        score_log = np.zeros((len(in_loader.dataset), num_classes))
        label_log = np.zeros(len(in_loader.dataset))

        model.eval()
        for batch_idx, (inputs, targets) in enumerate(in_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            start_ind = batch_idx * batch_size
            end_ind = min((batch_idx + 1) * batch_size, len(in_loader.dataset))

            score, feature_list = model.feature_list(inputs)
            out = torch.cat([F.adaptive_avg_pool2d(layer_feat, 1).squeeze() for layer_feat in feature_list], dim=1)

            feat_log[start_ind:end_ind, :] = out.data.cpu().numpy()
            label_log[start_ind:end_ind] = targets.data.cpu().numpy()
            score_log[start_ind:end_ind] = score.data.cpu().numpy()
            if batch_idx % 100 == 0:
                print(f"{batch_idx}/{len(in_loader)}")
        np.save(cache_name, (feat_log.T, score_log.T, label_log))
    else:
        feat_log, score_log, label_log = np.load(cache_name, allow_pickle=True)
        feat_log, score_log = feat_log.T, score_log.T

for ood_dataset in args.out_datasets:
    print(ood_dataset)
    
    loader_test_dict = get_loader_out(args, dataset=(None, ood_dataset), split=('val'))
    out_loader = loader_test_dict.val_ood_loader
    cache_name = f"cache/{ood_dataset}vs{args.in_dataset}_{args.name}_out_alllayers.npy"
    if FORCE_RUN or not os.path.exists(cache_name):
        ood_feat_log = np.zeros((len(out_loader.dataset), sum(featdims)))
        ood_score_log = np.zeros((len(out_loader.dataset), num_classes))

        model.eval()
        for batch_idx, (inputs, _) in enumerate(out_loader):
            inputs = inputs.to(device)
            start_ind = batch_idx * batch_size
            end_ind = min((batch_idx + 1) * batch_size, len(out_loader.dataset))

            score, feature_list = model.feature_list(inputs)
            out = torch.cat([F.adaptive_avg_pool2d(layer_feat, 1).squeeze() for layer_feat in feature_list], dim=1)

            ood_feat_log[start_ind:end_ind, :] = out.data.cpu().numpy()
            ood_score_log[start_ind:end_ind] = score.data.cpu().numpy()
            if batch_idx % 100 == 0:
                print(f"{batch_idx}/{len(out_loader)}")
        np.save(cache_name, (ood_feat_log.T, ood_score_log.T))
    else:
        ood_feat_log, ood_score_log = np.load(cache_name, allow_pickle=True)
        ood_feat_log, ood_score_log = ood_feat_log.T, ood_score_log.T
print(time.time() - begin)





#*********************************************************************************************************************

def get_one_model_p_value(ftrain, ftest, food, model_name, ood_dataset, K):

    #################### KNN score OOD detection #################

    index = faiss.IndexFlatL2(ftrain.shape[1])
    index.add(ftrain)


    D, _ = index.search(ftest, K)
    scores_in = -D[:,-1]
    all_results = []
    D, _ = index.search(food, K)
    scores_ood_test = -D[:,-1]
    results, in_p, out_p = metrics.cal_p_value(scores_in, scores_ood_test)
    all_results.append(results)

    return in_p, out_p, results  # array([num_in,]), array([num_out,])


def reload_feat(path, model_name):

    if 'train' in path or 'val' in path:
        feat_log, score_log, label_log = np.load(path, allow_pickle=True)
    else:
        feat_log, score_log = np.load(path, allow_pickle=True)
    feat_log, score_log = feat_log.T.astype(np.float32), score_log.T.astype(np.float32)
    class_num = score_log.shape[1]

    normalizer = lambda x: x / (np.linalg.norm(x, ord=2, axis=-1, keepdims=True) + 1e-10)
    if model_name.startswith('resnet'):
        model_num = int(''.join([x for x in model_name if x.isdigit()]))
        dim = feat_log.shape[1]
        if model_num < 50:
            # print(model_name, model_num, dim, 'if')
            #prepos_feat = lambda x: np.ascontiguousarray(normalizer(x[:, range(dim - 512, dim)]))  # Last Layer only
            prepos1_feat = lambda x: np.ascontiguousarray(normalizer(x[:, range(0,64)]))
            prepos2_feat = lambda x: np.ascontiguousarray(normalizer(x[:, range(64,192)]))
            prepos3_feat = lambda x: np.ascontiguousarray(normalizer(x[:, range(192,448)]))
            prepos4_feat = lambda x: np.ascontiguousarray(normalizer(x[:, range(448,960)]))
            pos_feat_log = [prepos1_feat(feat_log),prepos2_feat(feat_log),prepos3_feat(feat_log),
                            prepos4_feat(feat_log)]
        else:
            # print(model_name, model_num, dim, 'else')
            #prepos_feat = lambda x: np.ascontiguousarray(normalizer(x[:, range(dim - 2048, dim)]))  # Last Layer only
            prepos1_feat = lambda x: np.ascontiguousarray(normalizer(x[:, range(0,256)]))
            prepos2_feat = lambda x: np.ascontiguousarray(normalizer(x[:, range(256,768)]))
            prepos3_feat = lambda x: np.ascontiguousarray(normalizer(x[:, range(768,1792)]))
            prepos4_feat = lambda x: np.ascontiguousarray(normalizer(x[:, range(1792,3840)]))
            pos_feat_log = [prepos1_feat(feat_log),prepos2_feat(feat_log),prepos3_feat(feat_log),
                            prepos4_feat(feat_log)]
            
    elif model_name.startswith('densenet'):
        dim = feat_log.shape[1]
        # print(model_name, dim)
        prepos_feat = lambda x: np.ascontiguousarray(normalizer(x[:, range(dim - 342, dim)]))  # Last Layer only
#     pos_feat_log = prepos_feat(feat_log)   # [num, 512 or 2048 or 342]
    return pos_feat_log


def save_file(data, file_name):
    with open(file_name, 'w') as file_obj:
        json.dump(data, file_obj)

        
        
FORCE_RUN = True
K=args.K
sys.stdout = Logger(os.path.join('llog', 'log of mult-test by knn when k={}.txt'.format(K)))
# model_zoo = vars(args)[f"{args.in_dataset.replace('-', '_')}_model_zoo"]
# for model_name in model_zoo:
model_name = args.name

flag = 1
all_results=[]
for ood_dataset in args.out_datasets:
    cache_name = f"cache/{ood_dataset}vs{args.in_dataset}_{model_name}_out_alllayers.npy"
    food = reload_feat(cache_name, model_name)
    for i in range(4):
        p_value_dir = f"p-value/{args.in_dataset}/{ood_dataset}/{model_name}/feature_{i+1}_p_value{K}.json"
        if FORCE_RUN or not os.path.exists(p_value_dir):
            if not os.path.exists(os.path.dirname(p_value_dir)):
                os.makedirs(os.path.dirname(p_value_dir))
            if flag:
                cache_name = f"cache/{args.in_dataset}_train_{model_name}_in_alllayers.npy"
                ftrain = reload_feat(cache_name, model_name)
                cache_name = f"cache/{args.in_dataset}_val_{model_name}_in_alllayers.npy"
                ftest = reload_feat(cache_name, model_name)
                p_dit = {}
                flag = 0
            
            in_p, out_p, results = get_one_model_p_value(ftrain[i], ftest[i], food[i], model_name, ood_dataset, K=K)
            all_results.append(results)
            p_dit['in_p'] = in_p.tolist()
            p_dit['out_p'] = out_p.tolist()
            save_file(p_dit, p_value_dir)
print(f'when model is {model_name},k={K},the result is:')
metrics.print_all_results(all_results, args.out_datasets, 'knn')





#***********************************************************************************************************

# def cal_k_lis(p_value, alpha=0.05):
    
#         num, dim = p_value.shape
#         range_lis = np.arange(1, dim + 1) * alpha / dim
#         comp_arr = np.array([range_lis] * num)
#         p_in_sort = np.sort(p_value)

#         comp_arr -= p_in_sort

#         res = np.where(comp_arr >= 0, 1, 0).sum(1)
#         where_id = np.where(res == 0)
#         where_ood = np.where(res >= 1)
#         res[where_id] = 1
#         res[where_ood] = 0
    
#     return res

def fpr_auc_multi(fpr_l,tpr_l):#(fpr_l,tpr_l):
    #==================FPR95计算=========================
    tpr95=0.95
    fpr0=0
    tpr0=0
    for i,(fpr1,tpr1) in enumerate(zip(fpr_l[::-1],tpr_l[::-1])):
        if tpr1>=tpr95:
            break
        fpr0=fpr1
        tpr0=tpr1
    fpr95 = ((tpr95-tpr0)*fpr1 + (tpr1-tpr95)*fpr0) / (tpr1-tpr0)
    auc_ = auc(fpr_l,tpr_l)

    return auc_,fpr95

def cal_k_auc_fpr95(p_value_in,p_value_ood,multi_method):
    p_value_in_ood = np.concatenate([p_value_ood,p_value_in])
    num_Ip, num_layer = p_value_in.shape
    num_Op, num_layer = p_value_ood.shape
    threshold = np.arange(0,1,0.01)
#     fpr_dict = dict([(w,[]) for w in range(num_layer)])
#     tpr_dict = dict([(w,[]) for w in range(num_layer)])
    fpr_l_BH,fpr_l_adaBH,fpr_l_BY,fpr_l_Fisher,fpr_l_Cauchy = [],[],[],[],[]
    tpr_l_BH,tpr_l_adaBH,tpr_l_BY,tpr_l_Fisher,tpr_l_Cauchy = [],[],[],[],[]
    labels = np.concatenate([np.zeros(num_Op),np.ones(num_Ip)], axis=0)
    
        
    for thre in threshold:
        #pre = dict([(w,[]) for w in range(num_layer)])
        BH_pre,adaBH_pre,BY_pre,Fisher_pre,Cauchy_pre = [],[],[],[],[]
        
        for i in range(num_Op+num_Ip):
            obs = list(np.sort(p_value_in_ood[i,:]))
            
            #=========多种多重检验方法======================================
            if multi_method=='BH' or multi_method=='all':
                logic = [obs[j] <= ((j+1)*thre/num_layer) for j in range(len(obs))]
                k = len(logic) - logic[::-1].index(True) - 1 if True in logic else -1
                if k!=-1:BH_pre.append(0)
                else:BH_pre.append(1)
            if multi_method=='adaBH' or multi_method=='all':
                logic = [obs[j] <= ((j+1)*thre/num_layer) for j in range(len(obs))]
                if True not in logic: adaBH_pre.append(1)
                else:
                    pi_BH = [(num_layer-j)/(1-obs[j]+1e-16) for j in range(num_layer)]
                    if list(np.argwhere(np.diff(np.array(pi_BH))>0)):
                        J = np.argwhere(np.diff(np.array(pi_BH))>0)[0][0]+2
                    else:J = 1
                    #logic = [obs[j] <= ((j+1)*thre/min(pi_BH[J-1]+1,num_layer)) for j in range(num_layer)]
                    logic = [obs[j] <= ((j+1)*thre/pi_BH[J-1]) for j in range(num_layer)]
                    if True in logic:adaBH_pre.append(0)
                    else:adaBH_pre.append(1)
            if multi_method=='BY' or multi_method=='all':
                logic = [obs[j] <= ((j+1)*thre/(num_layer*(1/np.arange(1,num_layer+1)).sum())) for j in range(len(obs))]
                if True in logic: BY_pre.append(0)
                else:BY_pre.append(1)
            if multi_method=='Fisher' or multi_method=='all':
                if (-2*np.log(np.array(obs)+1e-16)).sum()>ss.chi2.ppf(1-thre,2*num_layer):
                    Fisher_pre.append(0)
                else:Fisher_pre.append(1)
            if multi_method=='Cauchy' or multi_method=='all':
                if np.tan((0.5-np.array(obs))*math.pi).sum()/num_layer > ss.cauchy.ppf(1-thre):
                    Cauchy_pre.append(0)
                else:Cauchy_pre.append(1)
                
                    
        if multi_method=='BH' or multi_method=='all':
            tpr_l_BH.append(((BH_pre+labels)==2).sum()/(labels==1).sum())
            fpr_l_BH.append(((BH_pre-labels)==1).sum()/(labels==0).sum())
        if multi_method=='adaBH' or multi_method=='all':
            tpr_l_adaBH.append(((adaBH_pre+labels)==2).sum()/(labels==1).sum())
            fpr_l_adaBH.append(((adaBH_pre-labels)==1).sum()/(labels==0).sum())
        if multi_method=='BY' or multi_method=='all':
            tpr_l_BY.append(((BY_pre+labels)==2).sum()/(labels==1).sum())
            fpr_l_BY.append(((BY_pre-labels)==1).sum()/(labels==0).sum())
        if multi_method=='Fisher' or multi_method=='all':
            tpr_l_Fisher.append(((Fisher_pre+labels)==2).sum()/(labels==1).sum())
            fpr_l_Fisher.append(((Fisher_pre-labels)==1).sum()/(labels==0).sum())
        if multi_method=='Cauchy' or multi_method=='all':
            tpr_l_Cauchy.append(((Cauchy_pre+labels)==2).sum()/(labels==1).sum())
            fpr_l_Cauchy.append(((Cauchy_pre-labels)==1).sum()/(labels==0).sum())
            
    if multi_method=='BH': 
        return fpr_auc_multi(BH_pre,labels)         #(fpr_l_BH,tpr_l_BH)
    if multi_method=='adaBH':
        return fpr_auc_multi(adaBH_pre,labels) 
    if multi_method=='BY':
        return fpr_auc_multi(BY_pre,labels) 
    if multi_method=='Fisher':
        return fpr_auc_multi(Fisher_pre,labels) 
    if multi_method=='Cauchy':
        return fpr_auc_multi(Cauchy_pre,labels) 
    if multi_method=='all':
        auc_all,fpr95_all = [],[]
        for fpr_l,tpr_l in zip([fpr_l_BH,fpr_l_adaBH,fpr_l_BY,fpr_l_Fisher,fpr_l_Cauchy],
                              [tpr_l_BH,tpr_l_adaBH,tpr_l_BY,tpr_l_Fisher,tpr_l_Cauchy]):
        #for Multi_pre in [BH_pre,adaBH_pre,BY_pre,Fisher_pre,Cauchy_pre]
            auc_,fpr95 = fpr_auc_multi(fpr_l,tpr_l) 
            auc_all.append(auc_)
            fpr95_all.append(fpr95)
        return auc_all,fpr95_all
    


def auroc(T_score, F_score):
    labels = np.concatenate([np.ones_like(T_score), np.zeros_like(F_score)], axis=0)
    scores = np.concatenate([T_score              , F_score               ], axis=0)
    return roc_auc_score(labels, scores)
def fpr95(T_score, F_score):
    tpr95=0.95
    labels = np.concatenate([np.ones_like(T_score), np.zeros_like(F_score)], axis=0)
    scores = np.concatenate([T_score              , F_score               ], axis=0)
    fpr,tpr,thresh = roc_curve(labels,scores)

    fpr0=0
    tpr0=0
    for i,(fpr1,tpr1) in enumerate(zip(fpr,tpr)):
        if tpr1>=tpr95:
            break
        fpr0=fpr1
        tpr0=tpr1
    fpr95 = ((tpr95-tpr0)*fpr1 + (tpr1-tpr95)*fpr0) / (tpr1-tpr0)
    return fpr95



def read_data(path):
    with open(path) as file_obj:
        p_value_dit = json.load(file_obj)
    return p_value_dit


def get_tp_fp_of_alpha(p_value_in, p_value_ood, alpha):
    id_pre_label = cal_k_lis(p_value_in, alpha)
    ood_pre_label = cal_k_lis(p_value_ood, alpha)
    tp = sum(id_pre_label)
    fp = sum(ood_pre_label)
    return (tp, fp)


def cal_auc(p_value_in, p_value_ood, space=2000):
    tp_fp = []
    for i in tqdm(range(0, space + 1)):
        alpha = 1 / space * i
        tp_fp.append(get_tp_fp_of_alpha(p_value_in, p_value_ood, alpha))

    tp_fp = sorted(tp_fp, reverse=True)
    tp_fp_arr = np.array(tp_fp).T

    tpr = np.concatenate(([[1.], tp_fp_arr[0] / p_value_in.shape[0], [0.]]))
    fpr = np.concatenate([[1.], tp_fp_arr[1] / p_value_ood.shape[0], [0.]])

    return auc(fpr, tpr)


def cal_auc_by_score(known, novel):
    num_k = known.shape[0]
    num_n = novel.shape[0]

    pred_score = np.concatenate((known, novel))
    true_label = np.zeros(num_k+num_n)
    true_label[:num_k] += 1
    auroc = roc_auc_score(true_label, pred_score)
    return auroc


def Multi(args, ood_dataset,method,k):
    p_value_in = []
    p_value_ood = []
    #model_zoo = vars(args)[f"{args.in_dataset.replace('-','_')}_model_zoo"]
    #for model_name in model_zoo:
    model_name = args.name
    auroc_l,fpr95_l = [],[]
    for i in range(4):
        if method == 'knn':
            p_value_dir = f"p-value/{args.in_dataset}/{ood_dataset}/{model_name}/feature_{i+1}_p_value{k}.json"
        else:
            p_value_dir = f"p-value/{method}/{args.in_dataset}/{ood_dataset}/{model_name}/p_value.json"
        p_value_dit = read_data(p_value_dir)
        p_value_in.append(p_value_dit['in_p'])
        p_value_ood.append(p_value_dit['out_p'])
        
        auroc_l.append(auroc(p_value_dit['in_p'], p_value_dit['out_p']))
        fpr95_l.append(fpr95(p_value_dit['in_p'], p_value_dit['out_p']))
    p_value_in = np.array(p_value_in).T
    p_value_ood = np.array(p_value_ood).T
    
    auroc_multi,fpr95_multi = cal_k_auc_fpr95(p_value_in,p_value_ood,multi_method=args.multi_method)
    return auroc_multi,fpr95_multi,auroc_l[-1],fpr95_l[-1]
        
        
        


method = 'knn'
sys.stdout = Logger(os.path.join('llog', 'log of cal res {} by {}, k={}.txt'.format(args.in_dataset,method,K)))


auroc_Last = []
fpr95_Last = []
auroc_Multi,fpr95_Multi = [],[]
for ood_dataset in args.out_datasets:
    auroc_multi,fpr95_multi,auroc_last,fpr95_last = Multi(args, ood_dataset,method,K)
    auroc_Last.append(auroc_last)
    fpr95_Last.append(fpr95_last)
    auroc_Multi.append(auroc_multi)
    fpr95_Multi.append(fpr95_multi)



print('********** ',args.model_arch,':auroc result ',args.in_dataset,' with ',method,' **********')


if args.multi_method != 'all':
        
    print('                              auroc                              fpr95    ')
    print('OOD dataset      exit@last        {}         exit@last       {}'.format(args.multi_method,
                                                                                                 args.multi_method))
    for i in range(len(args.out_datasets)):
        data_name=args.out_datasets[i]
        data_name = data_name + ' '*(17-len(data_name))
        print(data_name,"%.4f"%auroc_Last[i],'   ',"%.4f"%auroc_Multi[i],'    ',"%.4f"%fpr95_Last[i],'   ',"%.4f"%fpr95_Multi[i])
    data_name = 'average'
    data_name = data_name + ' '*(17-len(data_name))
    print(data_name,"%.4f"%np.mean(auroc_Last),'   ',"%.4f"%np.mean(auroc_Multi),'    ',
          "%.4f"%np.mean(fpr95_Last),'   ',"%.4f"%np.mean(fpr95_Multi))
else:
    
    print('                                       ****auroc****                              ')
    print('OOD dataset       exit@last     BH        adaBH      BY       Fisher      Cauchy')
    for i in range(len(args.out_datasets)):
        data_name=args.out_datasets[i]
        data_name = data_name + ' '*(17-len(data_name))
        print(data_name,"%.4f"%auroc_Last[i],'    ',"%.4f"%auroc_Multi[i][0],'    ',"%.4f"%auroc_Multi[i][1],'   ',"%.4f"%auroc_Multi[i][2],'   ',"%.4f"%auroc_Multi[i][3],'   ',"%.4f"%auroc_Multi[i][4])
    data_name = 'average'
    data_name = data_name + ' '*(17-len(data_name))
    print(data_name,"%.4f"%np.mean(auroc_Last),'    ',"%.4f"%np.mean([i[0] for i in auroc_Multi]),'    ',
          "%.4f"%np.mean([i[1] for i in auroc_Multi]),'   ',"%.4f"%np.mean([i[2] for i in auroc_Multi]),
          '    ',"%.4f"%np.mean([i[3] for i in auroc_Multi]),'    ',"%.4f"%np.mean([i[4] for i in auroc_Multi]))
    
    print('------------------------------------------------------------------------------------------------')
    print('                                       ****fpr95****                              ')
    print('OOD dataset       exit@last      BH        adaBH      BY       Fisher      Cauchy')
    for i in range(len(args.out_datasets)):
        data_name=args.out_datasets[i]
        data_name = data_name + ' '*(17-len(data_name))
        print(data_name,"%.4f"%fpr95_Last[i],'    ',"%.4f"%fpr95_Multi[i][0],'    ',"%.4f"%fpr95_Multi[i][1],'   ',"%.4f"%fpr95_Multi[i][2],'   ',"%.4f"%fpr95_Multi[i][3],'   ',"%.4f"%fpr95_Multi[i][4])
    data_name = 'average'
    data_name = data_name + ' '*(17-len(data_name))
    print(data_name,"%.4f"%np.mean(fpr95_Last),'    ',"%.4f"%np.mean([i[0] for i in fpr95_Multi]),'    ',
          "%.4f"%np.mean([i[1] for i in fpr95_Multi]),'   ',"%.4f"%np.mean([i[2] for i in fpr95_Multi]),
          '    ',"%.4f"%np.mean([i[3] for i in fpr95_Multi]),'    ',"%.4f"%np.mean([i[4] for i in fpr95_Multi]))
    



