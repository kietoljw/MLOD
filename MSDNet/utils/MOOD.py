import torch
import os

import cv2
import pickle
import numpy as np
import time

from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import auc
import scipy.stats as ss
import math

def calculate_complex(batch_data,NM):
    MEAN = NM[0]
    STD = NM[1]
    MEAN= torch.from_numpy(np.array(MEAN)).view([3,1,1])
    STD = torch.from_numpy(np.array(STD)).view([3,1,1])
    
    S = batch_data.shape
    complexity = np.zeros([S[0],1])
    for i in range(S[0]):
        img = batch_data[i]
        img = (img*STD+MEAN)*255.
        img = img.numpy().transpose([1,2,0])
        img = np.round(img).astype('uint8')
        cv2.imwrite('compressed.png', img, [cv2.IMWRITE_PNG_COMPRESSION , 9])
        complexity[i] = os.path.getsize('compressed.png')
    return complexity

def msp_score(pres, TF, L):
    for i in range(L):
        scores = np.max(F.softmax(pres[i], dim=1).detach().cpu().numpy(), axis=1)
        TF[i].append(scores)
    return scores

def energy_score(pres, TF, L, T=1):
    for i in range(L):
        scores  = T*torch.log( torch.sum( torch.exp(pres[i].detach().cpu().type(torch.DoubleTensor) / T), dim=1)).numpy()
        TF[i].append(scores)
    return scores

def odin_score(inputs, TF, model, L, temper=1000, noiseMagnitude=0.001):
    for i in range(L):
        criterion = nn.CrossEntropyLoss()
        inputs = Variable(inputs, requires_grad = True)
        inputs = inputs.cuda()
        inputs.retain_grad()
        
        outputs = model(inputs)[0][i]
        
        maxIndexTemp = np.argmax(outputs.data.cpu().numpy(), axis=1)
        
        # Using temperature scaling
        outputs = outputs / temper
        
        labels = Variable(torch.LongTensor(maxIndexTemp).cuda())
        loss = criterion(outputs, labels)
        loss.backward()
    
        # Normalizing the gradient to binary in {0, 1}
        gradient =  torch.ge(inputs.grad.data, 0)
        gradient = (gradient.float() - 0.5) * 2
    
        # Adding small perturbations to images
        tempInputs = torch.add(inputs.data, gradient, alpha=-noiseMagnitude)
        outputs = model(Variable(tempInputs))[0][i]
        outputs = outputs / temper
        # Calculating the confidence after adding perturbations
        nnOutputs = outputs.data.cpu()
        nnOutputs = nnOutputs.numpy()
        nnOutputs = nnOutputs - np.max(nnOutputs, axis=1, keepdims=True)
        nnOutputs = np.exp(nnOutputs) / np.sum(np.exp(nnOutputs), axis=1, keepdims=True)
        scores = np.max(nnOutputs, axis=1)
        
        TF[i].append(scores)
    return scores

def sample_estimator(model, num_classes, feature_list, data_loader):
    """
    compute sample mean and precision (inverse of covariance)
    return: sample_class_mean: list of class mean
             precision: list of precisions
    """
    import sklearn.covariance

    model.eval()
    group_lasso = sklearn.covariance.EmpiricalCovariance(assume_centered=False)
    correct, total = 0, 0
    num_output = len(feature_list)
    num_sample_per_class = np.empty(num_classes)
    num_sample_per_class.fill(0)
    list_features = []
    for i in range(num_output):
        temp_list = []
        for j in range(num_classes):
            temp_list.append(0)
        list_features.append(temp_list)
    
    for W in range(1):
        for data, target in data_loader:
            total += data.size(0)
            data = Variable(data)
            data = data.cuda()
            output, out_features = model(data)
            
            
            # get hidden features
            for i in range(num_output):
                out_features[i] = out_features[i].view(out_features[i].size(0), out_features[i].size(1), -1)
                out_features[i] = torch.mean(out_features[i].data, 2)
                
            # compute the accuracy
            output = output[-1]      
            pred = output.data.max(1)[1]
            equal_flag = pred.eq(target.cuda()).cpu()
            correct += equal_flag.sum()
    
            # construct the sample matrix
            for i in range(data.size(0)):
                label = target[i]
                if num_sample_per_class[label] == 0:
                    out_count = 0
                    for out in out_features:
                        list_features[out_count][label] = out[i].view(1, -1)
                        out_count += 1
                else:
                    out_count = 0
                    for out in out_features:
                        list_features[out_count][label] \
                        = torch.cat((list_features[out_count][label], out[i].view(1, -1)), 0)
                        out_count += 1
                num_sample_per_class[label] += 1

    sample_class_mean = []
    out_count = 0
    for num_feature in feature_list:
        temp_list = torch.Tensor(num_classes, int(num_feature)).cuda()
        for j in range(num_classes):
            temp_list[j] = torch.mean(list_features[out_count][j], 0)
        sample_class_mean.append(temp_list)
        out_count += 1

    precision = []
    for k in range(num_output):
        X = 0
        for i in range(num_classes):
            if i == 0:
                X = list_features[k][i] - sample_class_mean[k][i]
            else:
                X = torch.cat((X, list_features[k][i] - sample_class_mean[k][i]), 0)

        # find inverse
        group_lasso.fit(X.cpu().numpy())
        temp_precision = group_lasso.precision_
        temp_precision = torch.from_numpy(temp_precision).float().cuda()
        precision.append(temp_precision)

    print('\n Training Accuracy:({:.2f}%)\n'.format(100. * correct / total))

    return sample_class_mean, precision

def mahalanobis_score(inputs, TF, model, L):
    data_input = open('mahalanobis_parameters/sample_mean.pkl','rb')
    sample_mean = pickle.load(data_input)
    data_input.close()
    data_input = open('mahalanobis_parameters/precision.pkl','rb')
    precision = pickle.load(data_input)
    data_input.close()
    data_input = open('mahalanobis_parameters/num_classes.pkl','rb')
    num_classes = pickle.load(data_input)
    data_input.close()
    data_input = open('mahalanobis_parameters/magnitude.pkl','rb')
    magnitude = pickle.load(data_input)
    data_input.close()
    for layer_index in range(L):
        data = Variable(inputs, requires_grad = True)
        data = data.cuda()
        data.retain_grad()

        out_features = model(data)[1][layer_index]

        out_features = out_features.view(out_features.size(0), out_features.size(1), -1)
        out_features = torch.mean(out_features, 2)

        gaussian_score = 0
        for i in range(num_classes):
            batch_sample_mean = sample_mean[layer_index][i]
            zero_f = out_features.data - batch_sample_mean
            term_gau = -0.5*torch.mm(torch.mm(zero_f, precision[layer_index]), zero_f.t()).diag()
            if i == 0:
                gaussian_score = term_gau.view(-1,1)
            else:
                gaussian_score = torch.cat((gaussian_score, term_gau.view(-1,1)), 1)

        # Input_processing
        sample_pred = gaussian_score.max(1)[1]
        batch_sample_mean = sample_mean[layer_index].index_select(0, sample_pred)
        zero_f = out_features - Variable(batch_sample_mean)
        pure_gau = -0.5*torch.mm(torch.mm(zero_f, Variable(precision[layer_index])), zero_f.t()).diag()
        loss = torch.mean(-pure_gau)
        loss.backward()
        
        gradient =  torch.ge(data.grad.data, 0)
        gradient = (gradient.float() - 0.5) * 2

        tempInputs = torch.add(data.data, gradient, alpha=-magnitude)

        noise_out_features = model(Variable(tempInputs))[1][layer_index]
        noise_out_features = noise_out_features.view(noise_out_features.size(0), noise_out_features.size(1), -1)
        noise_out_features = torch.mean(noise_out_features, 2)
        noise_gaussian_score = 0
        for i in range(num_classes):
            batch_sample_mean = sample_mean[layer_index][i]
            zero_f = noise_out_features.data - batch_sample_mean
            term_gau = -0.5*torch.mm(torch.mm(zero_f, precision[layer_index]), zero_f.t()).diag()
            if i == 0:
                noise_gaussian_score = term_gau.view(-1,1)
            else:
                noise_gaussian_score = torch.cat((noise_gaussian_score, term_gau.view(-1,1)), 1)

        noise_gaussian_score, _ = torch.max(noise_gaussian_score, dim=1)

        noise_gaussian_score = np.asarray(noise_gaussian_score.cpu().numpy(), dtype=np.float32)
        if layer_index == 0:
            Mahalanobis_scores = noise_gaussian_score.reshape((noise_gaussian_score.shape[0], -1))
        else:
            Mahalanobis_scores = np.concatenate((Mahalanobis_scores, noise_gaussian_score.reshape((noise_gaussian_score.shape[0], -1))), axis=1)
    
    for i in range(L):
        TF[i].append(Mahalanobis_scores[:, i])
    return Mahalanobis_scores

def cut_transfer(L, threshold, energy, complexity, mean):
    cut_score = []
    for i in range(L):
        index = (threshold[i]<complexity) * (complexity<=threshold[i+1])
        index = index.reshape([-1])
        cut_score.append(energy[i][index]-mean[i])
    cut_score = np.concatenate(cut_score)
    return cut_score

def get_ood_score(data_name, model, L, dataloader, score_type, threshold, NM, 
                  adjusted_mode=0, mean=None, cal_complexity=True,T=1):
    
    score=[]
    if cal_complexity==True:
        complexity=[]
  
    for i in range(L):
        score.append([])
        
    num=0
    for images, labels in dataloader:

        if cal_complexity==True:
            complexity.append(calculate_complex(images,NM))

        images = images.cuda()

        if score_type == 'energy':
            with torch.no_grad():
                pres, _ = model(images)
            energy_score(pres, score, L,T=T)  #修改超参数T
        elif score_type == 'msp':
            with torch.no_grad():
                pres, _ = model(images)
            msp_score(pres, score, L)
        elif score_type == 'odin':
            model.eval()
            odin_score(images, score, model, L)
        elif score_type == 'mahalanobis':
            model.eval()
            mahalanobis_score(images, score, model, L)

        num+=images.shape[0]
        
    score = [np.concatenate(x) for x in score]
    
    if cal_complexity==True:
        complexity = np.concatenate(complexity)
        np.save('complexity/'+data_name+'.npy',complexity)
    if cal_complexity==False:
        complexity=np.load('complexity/'+data_name+'.npy')
    
    if adjusted_mode==1:
        adjusted_score = cut_transfer(L, threshold, score, complexity, mean)
    elif adjusted_mode==0:
        adjusted_score = cut_transfer(L, threshold, score, complexity, [0,0,0,0,0,0])#修改部分
    else:
        print('Adjusted_score wrong! It can only be 0 or 1!')
    return score, adjusted_score, complexity


from sklearn.metrics import average_precision_score,roc_auc_score,roc_curve
def aupr (T_score, F_score):
    labels = np.concatenate([np.ones_like(T_score), np.zeros_like(F_score)], axis=0)
    scores = np.concatenate([T_score              , F_score               ], axis=0)
    return average_precision_score(labels, scores)
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





#========添加部分=========================================================================
def fpr_auc_multi(fpr_l,tpr_l):
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


def get_curve_multi(O_pvalue_l,I_pvalue_l,num_layer,method):
    threshold = np.arange(0,1,0.01)
#     fpr_dict = dict([(w,[]) for w in range(num_layer)])
#     tpr_dict = dict([(w,[]) for w in range(num_layer)])
    fpr_l_BH,fpr_l_adaBH,fpr_l_BY,fpr_l_Fisher,fpr_l_Cauchy = [],[],[],[],[]
    tpr_l_BH,tpr_l_adaBH,tpr_l_BY,tpr_l_Fisher,tpr_l_Cauchy = [],[],[],[],[]
    labels = np.concatenate([np.zeros_like(O_pvalue_l[0]),np.ones_like(I_pvalue_l[0])], axis=0)
    
        
    for thre in threshold:
        #pre = dict([(w,[]) for w in range(num_layer)])
        BH_pre,adaBH_pre,BY_pre,Fisher_pre,Cauchy_pre = [],[],[],[],[]
        num_Op = len(O_pvalue_l[0])
        num_Ip = len(I_pvalue_l[0])
        
        for i in range(num_Op+num_Ip):
            if i <= (num_Op-1):
                obs = {layer:O_pvalue_l[layer][i] for layer in range(num_layer)}
            else:
                obs = {layer:I_pvalue_l[layer][i-num_Op] for layer in range(num_layer)}
            obs = sorted(obs.values())
            
            #=========多种多重检验方法======================================
            if method=='BH' or method=='all':
                logic = [obs[j] <= ((j+1)*thre/num_layer) for j in range(len(obs))]
                k = len(logic) - logic[::-1].index(True) - 1 if True in logic else -1
                if k!=-1:BH_pre.append(0)
                else:BH_pre.append(1)
            if method=='adaBH' or method=='all':
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
            if method=='BY' or method=='all':
                logic = [obs[j] <= ((j+1)*thre/(num_layer*(1/np.arange(1,num_layer+1)).sum())) for j in range(len(obs))]
                if True in logic: BY_pre.append(0)
                else:BY_pre.append(1)
            if method=='Fisher' or method=='all':
                if (-2*np.log(obs)).sum()>ss.chi2.ppf(1-thre,2*num_layer):
                    Fisher_pre.append(0)
                else:Fisher_pre.append(1)
            if method=='Cauchy' or method=='all':
                if np.tan((0.5-np.array(obs))*math.pi).sum()/num_layer > ss.cauchy.ppf(1-thre):
                    Cauchy_pre.append(0)
                else:Cauchy_pre.append(1)
                
                    
        if method=='BH' or method=='all':
            tpr_l_BH.append(((BH_pre+labels)==2).sum()/(labels==1).sum())
            fpr_l_BH.append(((BH_pre-labels)==1).sum()/(labels==0).sum())
        if method=='adaBH' or method=='all':
            tpr_l_adaBH.append(((adaBH_pre+labels)==2).sum()/(labels==1).sum())
            fpr_l_adaBH.append(((adaBH_pre-labels)==1).sum()/(labels==0).sum())
        if method=='BY' or method=='all':
            tpr_l_BY.append(((BY_pre+labels)==2).sum()/(labels==1).sum())
            fpr_l_BY.append(((BY_pre-labels)==1).sum()/(labels==0).sum())
        if method=='Fisher' or method=='all':
            tpr_l_Fisher.append(((Fisher_pre+labels)==2).sum()/(labels==1).sum())
            fpr_l_Fisher.append(((Fisher_pre-labels)==1).sum()/(labels==0).sum())
        if method=='Cauchy' or method=='all':
            tpr_l_Cauchy.append(((Cauchy_pre+labels)==2).sum()/(labels==1).sum())
            fpr_l_Cauchy.append(((Cauchy_pre-labels)==1).sum()/(labels==0).sum())
            
    if method=='BH': 
        return fpr_auc_multi(fpr_l_BH,tpr_l_BH)
    if method=='adaBH':
        return fpr_auc_multi(fpr_l_adaBH,tpr_l_adaBH)
    if method=='BY':
        return fpr_auc_multi(fpr_l_BY,tpr_l_BY)
    if method=='Fisher':
        return fpr_auc_multi(fpr_l_Fisher,tpr_l_Fisher)
    if method=='Cauchy':
        return fpr_auc_multi(fpr_l_Cauchy,tpr_l_Cauchy)
    if method=='all':
        auc_all,fpr95_all = [],[]
        for fpr_l,tpr_l in zip([fpr_l_BH,fpr_l_adaBH,fpr_l_BY,fpr_l_Fisher,fpr_l_Cauchy],
                              [tpr_l_BH,tpr_l_adaBH,tpr_l_BY,tpr_l_Fisher,tpr_l_Cauchy]):
            auc_,fpr95 = fpr_auc_multi(fpr_l,tpr_l)
            auc_all.append(auc_)
            fpr95_all.append(fpr95)
        return auc_all,fpr95_all
    
    

#BH多重检验方法
#def auroc_BH(fpr_l,tpr_l,num_layer):
#     for layer in range(num_layer):
#         auc_dict[layer] = auc(fpr_dict[layer],tpr_dict[layer])
#     return auc(fpr_l,tpr_l)
            
        
# def fpr95_BH(fpr_l,tpr_l,num_layer):
#     tpr95=0.95
#     #fpr95_dict = dict()
#     #for layer in range(num_layer):
#     fpr0=0
#     tpr0=0
#     for i,(fpr1,tpr1) in enumerate(zip(fpr_l[::-1],tpr_l[::-1])):
#         if tpr1>=tpr95:
#             break
#         fpr0=fpr1
#         tpr0=tpr1
#     fpr95 = ((tpr95-tpr0)*fpr1 + (tpr1-tpr95)*fpr0) / (tpr1-tpr0)
#     return fpr95


        
#         auroc_l = list(auroc_BH(fpr_dict,tpr_dict,num_layer).values())
#         fpr95_l = list(fpr95_BH(fpr_dict,tpr_dict,num_layer).values())
#         auroc_l = list(auroc_BH(fpr_dict,tpr_dict,num_layer).values())
#========================================================================================