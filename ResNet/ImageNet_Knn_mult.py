import torch
import time
from util import metrics
import faiss
import os

from util.args_loader import get_args
from util.data_loader import get_loader_in, get_loader_out
from util.model_loader import get_model
import numpy as np
import torch.nn.functional as F
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


print(time.ctime())


torch.manual_seed(1)
torch.cuda.manual_seed(1)
np.random.seed(1)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

args = get_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

loader_in_dict = get_loader_in(args, config_type="eval", split=('train', 'val'))
trainloaderIn, testloaderIn, num_classes = loader_in_dict.train_loader, loader_in_dict.val_loader, loader_in_dict.num_classes
model = get_model(args, num_classes, load_ckpt=True)

batch_size = args.batch_size
featdim = {
    'resnet50_imagenet': 3840,
    'resnet34_imagenet': 960,
    'resnet18_imagenet': 960,
    'resnet50-supcon': 3840,#5888,
    'resnet101_imagenet': 3840,#5888,
    'resnet152_imagenet': 3840#5888,
}[args.model_arch]

FORCE_RUN = False
ID_RUN = True
OOD_RUN = True

if ID_RUN:
    for split, in_loader in [('val', testloaderIn), ('train', trainloaderIn)]:

        cache_dir = f"cache/{args.in_dataset}_{split}_{args.name}_in"
        if FORCE_RUN or not os.path.exists(cache_dir):
            os.makedirs(cache_dir, exist_ok=True)
            feat_log = np.memmap(f"{cache_dir}/feat.mmap", dtype=float, mode='w+', shape=(len(in_loader.dataset), featdim))
#             score_log = np.memmap(f"{cache_dir}/score.mmap", dtype=float, mode='w+', shape=(len(in_loader.dataset), num_classes))
            label_log = np.memmap(f"{cache_dir}/label.mmap", dtype=float, mode='w+', shape=(len(in_loader.dataset),))

            model.eval()
            for batch_idx, (inputs, targets) in enumerate(in_loader):

                inputs, targets = inputs.to(device), targets.to(device)
                start_ind = batch_idx * batch_size
                end_ind = min((batch_idx + 1) * batch_size, len(in_loader.dataset))

                feature_list = model.feature_list(inputs)
                Out_list = []
                for out in feature_list:
                    if len(out.shape) > 2:
                        out = F.adaptive_avg_pool2d(out, 1)
                        # out = out.clip(max=1.0)
                        out = out.view(out.size(0), -1)
                    Out_list.append(out)
                out = torch.cat(Out_list,dim=1)
                feat_log[start_ind:end_ind, :] = out.data.cpu().numpy()
                label_log[start_ind:end_ind] = targets.data.cpu().numpy()
#                 score_log[start_ind:end_ind] = score.data.cpu().numpy()
                if batch_idx % 100 == 0:
                    print(f"{batch_idx}/{len(in_loader)}")
        else:
            feat_log = np.memmap(f"{cache_dir}/feat.mmap", dtype=float, mode='r', shape=(len(in_loader.dataset), featdim))
#             score_log = np.memmap(f"{cache_dir}/score.mmap", dtype=float, mode='r', shape=(len(in_loader.dataset), num_classes))
            label_log = np.memmap(f"{cache_dir}/label.mmap", dtype=float, mode='r', shape=(len(in_loader.dataset),))

if OOD_RUN:

    for ood_dataset in args.out_datasets:
        print(ood_dataset)
        loader_test_dict = get_loader_out(args, dataset=(None, ood_dataset), split=('val'))
        out_loader = loader_test_dict.val_ood_loader
        print(len(out_loader.dataset))
        cache_dir = f"cache/{ood_dataset}vs{args.in_dataset}_{args.name}_out"
        if FORCE_RUN or not os.path.exists(cache_dir):
            os.makedirs(cache_dir, exist_ok=True)
            ood_feat_log = np.memmap(f"{cache_dir}/feat.mmap", dtype=float, mode='w+', shape=(len(out_loader.dataset), featdim))
#             ood_score_log = np.memmap(f"{cache_dir}/score.mmap", dtype=float, mode='w+', shape=(len(out_loader.dataset), num_classes))
            model.eval()
            for batch_idx, (inputs, _) in enumerate(out_loader):
                inputs = inputs.to(device)
                start_ind = batch_idx * batch_size
                end_ind = min((batch_idx + 1) * batch_size, len(out_loader.dataset))

                feature_list = model.feature_list(inputs)
                Out_list = []
                for out in feature_list:
                    if len(out.shape) > 2:
                        out = F.adaptive_avg_pool2d(out, 1)
                        # out = out.clip(max=1.0)
                        out = out.view(out.size(0), -1)
                    Out_list.append(out)
                out = torch.cat(Out_list,dim=1)
                ood_feat_log[start_ind:end_ind, :] = out.data.cpu().numpy()
#                 ood_score_log[start_ind:end_ind] = score.data.cpu().numpy()
                if batch_idx % 100 == 0:
                    print(f"{batch_idx}/{len(out_loader)}")


        else:
            ood_feat_log = np.memmap(f"{cache_dir}/feat.mmap", dtype=float, mode='r', shape=(len(out_loader.dataset), featdim))
#             ood_score_log = np.memmap(f"{cache_dir}/score.mmap", dtype=float, mode='r', shape=(len(out_loader.dataset), num_classes))
            
            
            
print("ID AND OOD DATASET RUN SUCCESSFUL!\n")            
            
            




def get_one_model_p_value(ftrain, ftest, food, model_name, ood_dataset, K):
    
    #################### KNN score OOD detection #################
#     ALPHA = 1.00
#     rand_ind = np.random.choice(id_train_size, int(id_train_size * ALPHA), replace=False)
    index = faiss.IndexFlatL2(ftrain.shape[1])
#     index.add(ftrain[rand_ind])
    index.add(ftrain)

    D, _ = index.search(ftest, K)
    scores_in = -D[:,-1]
    # all_results = []
    D, _ = index.search(food, K)
    scores_ood_test = -D[:,-1]
    results, in_p, out_p = metrics.cal_p_value(scores_in, scores_ood_test)
    # all_results.append(results)

    return in_p, out_p, results  # array([num_in,]), array([num_out,])

    
    
    

def reload_feat(feat_log, model_name):
    print(feat_log.shape)
    class_num = 1000
    if model_name.startswith('resnet'):
        model_num = int(''.join([x for x in model_name if x.isdigit()]))
        dim = feat_log.shape[1]
        if model_num < 50:
            # print(model_name, model_num, dim, 'if')
            #prepos_feat = lambda x: np.ascontiguousarray(normalizer(x[:, range(dim - 512, dim)]))  # Last Layer only
            prepos1_feat = lambda x: np.ascontiguousarray(normalizer(x[:, range(0,64)]).astype(np.float32))
            prepos2_feat = lambda x: np.ascontiguousarray(normalizer(x[:, range(64,192)]).astype(np.float32))
            prepos3_feat = lambda x: np.ascontiguousarray(normalizer(x[:, range(192,448)]).astype(np.float32))
            prepos4_feat = lambda x: np.ascontiguousarray(normalizer(x[:, range(448,960)]).astype(np.float32))
            pos_feat_log = [prepos1_feat(feat_log),prepos2_feat(feat_log),prepos3_feat(feat_log),
                            prepos4_feat(feat_log)]
        else:
            # print(model_name, model_num, dim, 'else')
            #prepos_feat = lambda x: np.ascontiguousarray(normalizer(x[:, range(dim - 2048, dim)]))  # Last Layer only
            prepos1_feat = lambda x: np.ascontiguousarray(normalizer(x[:, range(0,256)]).astype(np.float32))
            prepos2_feat = lambda x: np.ascontiguousarray(normalizer(x[:, range(256,768)]).astype(np.float32))
            prepos3_feat = lambda x: np.ascontiguousarray(normalizer(x[:, range(768,1792)]).astype(np.float32))
            prepos4_feat = lambda x: np.ascontiguousarray(normalizer(x[:, range(1792,3840)]).astype(np.float32))
#             prepos5_feat = lambda x: np.ascontiguousarray(normalizer(x[:, range(3840,5888)]).astype(np.float32))
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


    
    


##加载特征==============================================================================================================
class_num = 1000
id_train_size = 1281167
id_val_size = 50000

cache_dir = f"cache/{args.in_dataset}_train_{args.name}_in"
feat_log = np.memmap(f"{cache_dir}/feat.mmap", dtype=float, mode='r', shape=(id_train_size, featdim))
# score_log = np.memmap(f"{cache_dir}/score.mmap", dtype=float, mode='r', shape=(id_train_size, class_num))
label_log = np.memmap(f"{cache_dir}/label.mmap", dtype=float, mode='r', shape=(id_train_size,))


cache_dir = f"cache/{args.in_dataset}_val_{args.name}_in"
feat_log_val = np.memmap(f"{cache_dir}/feat.mmap", dtype=float, mode='r', shape=(id_val_size, featdim))
# score_log_val = np.memmap(f"{cache_dir}/score.mmap", dtype=float, mode='r', shape=(id_val_size, class_num))
label_log_val = np.memmap(f"{cache_dir}/label.mmap", dtype=float, mode='r', shape=(id_val_size,))

normalizer = lambda x: x / (np.linalg.norm(x,axis=-1, keepdims=True) + 1e-10)
FORCE_RUN = False
norm_cache = f"cache/{args.in_dataset}_train_{args.name}_in/feat_norm.mmap"
if not FORCE_RUN and os.path.exists(norm_cache):
    feat_log_norm = np.memmap(norm_cache, dtype=float, mode='r', shape=(id_train_size, featdim))
    
elif args.model_arch in ['resnet50_imagenet','resnet50-supcon']:
    feat_log_norm = np.memmap(norm_cache, dtype=float, mode='w+', shape=(id_train_size, featdim))
    feat_log_norm[:,range(0,256)] = normalizer(feat_log[:,range(0,256)])
    feat_log_norm[:,range(256,768)] = normalizer(feat_log[:,range(256,768)])
    feat_log_norm[:,range(768,1792)] = normalizer(feat_log[:,range(768,1792)])
    feat_log_norm[:,range(1792,3840)] = normalizer(feat_log[:,range(1792,3840)])
#     feat_log_norm[:,range(3840,5888)] = normalizer(feat_log[:,range(3840,5888)])
else:
    feat_log_norm = np.memmap(norm_cache, dtype=float, mode='w+', shape=(id_train_size, featdim))
    feat_log_norm[:,range(0,64)] = normalizer(feat_log[:,range(0,64)])
    feat_log_norm[:,range(64,192)] = normalizer(feat_log[:,range(64,192)])
    feat_log_norm[:,range(192,448)] = normalizer(feat_log[:,range(192,448)])
    feat_log_norm[:,range(448,960)] = normalizer(feat_log[:,range(448,960)])

    

norm_cache = f"cache/{args.in_dataset}_val_{args.name}_in/feat_norm.mmap"
if not FORCE_RUN and os.path.exists(norm_cache):
    feat_log_val_norm = np.memmap(norm_cache, dtype=float, mode='r', shape=(id_val_size, featdim))
elif args.model_arch in ['resnet50_imagenet','resnet50-supcon']:
    feat_log_val_norm = np.memmap(norm_cache, dtype=float, mode='w+', shape=(id_val_size, featdim))
    feat_log_val_norm[:,range(0,256)] = normalizer(feat_log_val[:,range(0,256)])
    feat_log_val_norm[:,range(256,768)] = normalizer(feat_log_val[:,range(256,768)])
    feat_log_val_norm[:,range(768,1792)] = normalizer(feat_log_val[:,range(768,1792)])
    feat_log_val_norm[:,range(1792,3840)] = normalizer(feat_log_val[:,range(1792,3840)])
#     feat_log_val_norm[:,range(3840,5888)] = normalizer(feat_log_val[:,range(3840,5888)])
else:
    feat_log_val_norm = np.memmap(norm_cache, dtype=float, mode='w+', shape=(id_val_size, featdim))
    feat_log_val_norm[:,range(0,64)] = normalizer(feat_log_val[:,range(0,64)])
    feat_log_val_norm[:,range(64,192)] = normalizer(feat_log_val[:,range(64,192)])
    feat_log_val_norm[:,range(192,448)] = normalizer(feat_log_val[:,range(192,448)])
    feat_log_val_norm[:,range(448,960)] = normalizer(feat_log_val[:,range(448,960)])
    
    
ftrain,ftest = [],[]
if args.model_arch in ['resnet50_imagenet','resnet50-supcon']:
    ftrain.append(np.ascontiguousarray(feat_log_norm[:, range(0,256)].astype(np.float32)))
    ftrain.append(np.ascontiguousarray(feat_log_norm[:, range(256,768)].astype(np.float32)))
    ftrain.append(np.ascontiguousarray(feat_log_norm[:, range(768,1792)].astype(np.float32)))
    ftrain.append(np.ascontiguousarray(feat_log_norm[:, range(1792,3840)].astype(np.float32)))
    # ftrain.append(np.ascontiguousarray(feat_log_norm[:, range(3840,5888)].astype(np.float32)))
    ftest.append(np.ascontiguousarray(feat_log_val_norm[:, range(0,256)].astype(np.float32)))
    ftest.append(np.ascontiguousarray(feat_log_val_norm[:, range(256,768)].astype(np.float32)))
    ftest.append(np.ascontiguousarray(feat_log_val_norm[:, range(768,1792)].astype(np.float32)))
    ftest.append(np.ascontiguousarray(feat_log_val_norm[:, range(1792,3840)].astype(np.float32)))
    # ftest.append(np.ascontiguousarray(feat_log_val_norm[:, range(3840,5888)].astype(np.float32)))
else:
    ftrain.append(np.ascontiguousarray(feat_log_norm[:,range(0,64)].astype(np.float32)))
    ftrain.append(np.ascontiguousarray(feat_log_norm[:, range(64,192)].astype(np.float32)))
    ftrain.append(np.ascontiguousarray(feat_log_norm[:, range(192,448)].astype(np.float32)))
    ftrain.append(np.ascontiguousarray(feat_log_norm[:, range(448,960)].astype(np.float32)))
    # ftrain.append(np.ascontiguousarray(feat_log_norm[:, range(3840,5888)].astype(np.float32)))
    ftest.append(np.ascontiguousarray(feat_log_val_norm[:, range(0,64)].astype(np.float32)))
    ftest.append(np.ascontiguousarray(feat_log_val_norm[:, range(64,192)].astype(np.float32)))
    ftest.append(np.ascontiguousarray(feat_log_val_norm[:, range(192,448)].astype(np.float32)))
    ftest.append(np.ascontiguousarray(feat_log_val_norm[:, range(448,960)].astype(np.float32)))
    # ftest.append(np.ascontiguousarray(feat_log_val_norm[:, range(3840,5888)].astype(np.float32)))
    
                     
                     
                     
ood_dataset_size = {
    'mnist':10000,
    'kmnist':10000,
    'fasionmnist':10000,
    'lsun':10000,
    'svhn':10000,
    'isun':20608,
    'lsunR':10000,
    'inat':10000,
    'sun50': 10000,
    'places50': 10000,
    'dtd': 5645
    }


##===============================================================================================================
FORCE_RUN = True
K=args.K
sys.stdout = Logger(os.path.join('llog', 'log of mult-test by knn when k={}.txt'.format(K)))
# model_zoo = vars(args)[f"{args.in_dataset.replace('-', '_')}_model_zoo"]
# for model_name in model_zoo:
model_name = args.name 
    
    
flag = 1
all_results=[]
for ood_dataset in args.out_datasets:
    print(ood_dataset)
    ood_feat_log = np.memmap(f"cache/{ood_dataset}vs{args.in_dataset}_{args.name}_out/feat.mmap", dtype=float, mode='r', shape=(ood_dataset_size[ood_dataset], featdim))
#     ood_score_log = np.memmap(f"cache/{ood_dataset}vs{args.in_dataset}_{args.name}_out/score.mmap", dtype=float, mode='r', shape=(ood_dataset_size[ood_dataset], class_num))
    
    food = reload_feat(ood_feat_log, model_name)
    for i in range(4):
        p_value_dir = f"p-value/{args.in_dataset}/{ood_dataset}/{model_name}/feature_{i+1}_p_value{K}.json"
        if FORCE_RUN or not os.path.exists(p_value_dir):
            if not os.path.exists(os.path.dirname(p_value_dir)):
                os.makedirs(os.path.dirname(p_value_dir))
            if flag:
                #ftrain = reload_idfeat(feat_log_norm, model_name)
                #ftest = reload_idfeat(feat_log_val_norm, model_name)
                p_dit = {}
                flag = 0
            
            in_p, out_p, results = get_one_model_p_value(ftrain[i], ftest[i], food[i], model_name, ood_dataset, K=K)
            all_results.append(results)
            p_dit['in_p'] = in_p.tolist()
            p_dit['out_p'] = out_p.tolist()
            save_file(p_dit, p_value_dir)
            
    print(f"{ood_dataset} P VALUE SUCCESSFUL!\n")      
# print(f'when model is {model_name},k={K},the result is:')
# print(len(all_results))
# metrics.print_all_results(all_results, args.out_datasets, 'knn')





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
    threshold = np.arange(0,1,0.001)
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
    print(data_name,"%.4f"%np.mean(auroc_Last),'    ',"%.4f"%np.mean([x[0] for x in auroc_Multi]),'    ',
          "%.4f"%np.mean([x[1] for x in auroc_Multi]),'   ',"%.4f"%np.mean([x[2] for x in auroc_Multi]),
          '    ',"%.4f"%np.mean([x[3] for x in auroc_Multi]),'    ',"%.4f"%np.mean([x[4] for x in auroc_Multi]))
    
    print('------------------------------------------------------------------------------------------------')
    print('                                       ****fpr95****                              ')
    print('OOD dataset       exit@last      BH        adaBH      BY       Fisher      Cauchy')
    for i in range(len(args.out_datasets)):
        data_name=args.out_datasets[i]
        data_name = data_name + ' '*(17-len(data_name))
        print(data_name,"%.4f"%fpr95_Last[i],'    ',"%.4f"%fpr95_Multi[i][0],'    ',"%.4f"%fpr95_Multi[i][1],'   ',"%.4f"%fpr95_Multi[i][2],'   ',"%.4f"%fpr95_Multi[i][3],'   ',"%.4f"%fpr95_Multi[i][4])
    data_name = 'average'
    data_name = data_name + ' '*(17-len(data_name))
    print(data_name,"%.4f"%np.mean(fpr95_Last),'    ',"%.4f"%np.mean([x[0] for x in fpr95_Multi]),'    ',
          "%.4f"%np.mean([x[1] for x in fpr95_Multi]),'   ',"%.4f"%np.mean([x[2] for x in fpr95_Multi]),
          '    ',"%.4f"%np.mean([x[3] for x in fpr95_Multi]),'    ',"%.4f"%np.mean([x[4] for x in fpr95_Multi]))
    


print(time.ctime())    



            