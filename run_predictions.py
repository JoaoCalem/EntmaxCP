import pandas as pd
from confpred.cp import run_cp
import numpy as np

alpha=0.1
#model_type = 'cnn'
#for dataset in ['MNIST','CIFAR10','CIFAR100']:
#    for loss in ['sparsemax','softmax','entmax']:
#        for seed in ['23','05','19','95','42']:
#            print(dataset+'_'+loss+'_'+seed+'_'+str(alpha))
#            avg_set_size, coverage = run_cp(dataset,loss,alpha,seed, model_type=model_type)

model_type = 'vit'
for dataset in ['CIFAR10','CIFAR100']:
    for loss in ['sparsemax','softmax','entmax']:
        for seed in ['05','19','95','42']:
            print(dataset+'_'+loss+'_'+seed+'_'+str(alpha))
            avg_set_size, coverage = run_cp(dataset,loss,alpha,seed, model_type=model_type, epochs=5)

""" model_type = 'bert'
for dataset in ['NewsGroups']:
    for loss in ['sparsemax','softmax','entmax']:
        for seed in ['23']:
            print(dataset+'_'+loss+'_'+seed+'_'+str(alpha))
            avg_set_size, coverage = run_cp(dataset,loss,alpha,seed, model_type=model_type)
 """