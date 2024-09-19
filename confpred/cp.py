from confpred import ConformalPredictor,SparseScore,SoftmaxScore
from confpred.classifier import CNN, evaluate
from confpred.datasets import CIFAR10, CIFAR100, MNIST

from entmax.losses import SparsemaxLoss, Entmax15Loss
import torch
import numpy as np
import pandas as pd 
import os.path
import pickle

def run_cp(dataset, loss, alpha, seed, model_type='cnn', epochs=20):
    #loss = 'softmax' #sparsemax, softmax or entmax15
    transformation = 'logits'
    #dataset='CIFAR100' #CIFAR100 or MNIST

    #device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device="cpu"

    n_class = 100 if dataset == 'CIFAR100' else 10
    if dataset in ['CIFAR100','CIFAR10']:
        model = CNN(n_class,
                    32,
                    3,
                    transformation=transformation,
                    conv_channels=[256,512,512],
                    convs_per_pool=2,
                    batch_norm=True,
                    ffn_hidden_size=1024,
                    kernel=5,
                    padding=2).to(device)
    if dataset == 'MNIST':
        model = CNN(10,
                    28,
                    1,
                    transformation=transformation).to(device)
        
    data_class = {
        'CIFAR100': CIFAR100,
        'CIFAR10': CIFAR10,
        'MNIST': MNIST,
    }

    data = data_class[dataset](0.2, 16, 3000, True)
    
    fname = f'./data/predictions/{model_type}_{dataset}_test_{loss}_{transformation}_{seed}_proba.pickle'
    if os.path.isfile(fname):
        print('Loading predictions.')
        path = f'./data/predictions/{model_type}_{dataset}_test_{loss}_{transformation}_{seed}_proba.pickle'
        with open(path, 'rb') as f:
            test_proba = pickle.load(f)
        path = f'./data/predictions/{dataset}_{seed}_test_true.pickle'
        with open(path, 'rb') as f:
            test_true_enc = pickle.load(f)
        path = f'./data/predictions/{model_type}_{dataset}_cal_{loss}_{transformation}_{seed}_proba.pickle'
        with open(path, 'rb') as f:
            cal_proba = pickle.load(f)
        path = f'./data/predictions/{dataset}_{seed}_cal_true.pickle'
        with open(path, 'rb') as f:
            cal_true_enc = pickle.load(f)
    else:
        model.load_state_dict(torch.load(f'./models/{model_type}_{dataset}_{loss}_{seed}_{epochs}_model.pth', map_location=torch.device(device)))
        print('Running predictions.')
        if loss == 'sparsemax':
            criterion = SparsemaxLoss()
        elif loss == 'softmax':
            criterion = torch.nn.NLLLoss()
        elif loss== 'entmax':
            criterion = Entmax15Loss()
            
        test_proba, _, test_true, _ = evaluate(
                                        model,
                                        data.test,
                                        criterion,
                                        True)

        cal_proba, _, cal_true, _ = evaluate(
                                        model,
                                        data.cal,
                                        criterion,
                                        True)

    #One Hot Encoding
        test_true_enc = np.zeros((test_true.size, test_true.max()+1), dtype=int)
        test_true_enc[np.arange(test_true.size),test_true] = 1

        cal_true_enc = np.zeros((cal_true.size, cal_true.max()+1), dtype=int)
        cal_true_enc[np.arange(cal_true.size),cal_true] = 1
        
        predictions = {'test':{'proba':test_proba,'true':test_true_enc},
                       'cal':{'proba':cal_proba,'true':cal_true_enc}}
        for dataset_type in ['cal','test']:
            with open(f'./data/predictions/{dataset}_{seed}_{dataset_type}_true.pickle', 'wb') as f:
                pickle.dump(predictions[dataset_type]['true'], f)
            with open(
                f'./data/predictions/{model_type}_{dataset}_{dataset_type}_{loss}' +
                    f'_{transformation}_{seed}_{"proba"}.pickle'
                , 'wb'
            ) as f:
                pickle.dump(predictions[dataset_type]["proba"], f)
    #Conformal Prediction
    if loss == 'sparsemax':
        cp = ConformalPredictor(SparseScore(2))
    elif loss == 'softmax':
        cp = ConformalPredictor(SoftmaxScore())
    elif loss== 'entmax':
        cp = ConformalPredictor(SparseScore(1.5))
    
    cp.calibrate(cal_true_enc, cal_proba, alpha)
    avg_set_size, coverage = cp.evaluate(test_true_enc, test_proba)
    
    return avg_set_size, coverage

