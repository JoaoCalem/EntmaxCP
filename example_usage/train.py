
from confpred.classifier import FineTuneViT, CNN, train, evaluate
from confpred.datasets import CIFAR10, CIFAR100, MNIST, ImageNet
from entmax.losses import SparsemaxLoss, Entmax15Loss
import json
import torch
from torch import nn
from sklearn.metrics import f1_score
import numpy as np
import random 
import os

seed = 123
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
# When running on the CuDNN backend, two further options must be set
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# Set a fixed value for the hash seed
os.environ["PYTHONHASHSEED"] = str(seed)
print(f"Random seed set as {seed}")

print('changed')

#loss = 'sparsemax' #sparsemax or softmax
#dataset = 'CIFAR10' #CIFARx =100 or MNIST
model_type = 'vit' #vit or cnn
for loss in ['sparsemax','entmax','softmax']:
    for dataset in ['CIFAR100']:
        print(loss, dataset)
        device = 'cuda:1' if torch.cuda.is_available() and torch.cuda.device_count()>1 else 'cuda' if torch.cuda.is_available() else 'cpu'
        print(device)

        if loss == 'sparsemax':
            criterion = SparsemaxLoss()
        elif loss == 'softmax':
            criterion = torch.nn.NLLLoss()
        elif loss == 'entmax':
            criterion = Entmax15Loss()

        data_class = {
            'ImageNet': ImageNet,
            'CIFAR100': CIFAR100,
            'CIFAR10': CIFAR10,
            'MNIST': MNIST,
        }
        if model_type=='vit':
            transforms = 'vit'
        else:
            transforms = 'norm'
        data = data_class[dataset](0.2, 8, 3000, transforms)


        n_class = 100 if dataset == 'CIFAR100' else 1000 if dataset == 'ImageNet' else 10
        input_size = 256 if dataset == 'ImageNet' else 32
        if model_type == 'cnn':
            if dataset in ['CIFAR100','CIFAR10','ImageNet']:
                model = CNN(n_class,
                            input_size,
                            3,
                            transformation=loss,
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
                            transformation=loss).to(device)
        if model_type == 'vit':
            model = FineTuneViT(n_class,transformation=loss).to(device)
            
        model, train_history, val_history, f1_history = train(model,
                                                    data.train,
                                                    data.dev,
                                                    criterion,
                                                    epochs=25,
                                                    patience=3)

        _, predicted_labels, true_labels, test_loss = evaluate(
                                                            model,
                                                            data.test,
                                                            criterion)

        f1 = f1_score(true_labels, predicted_labels, average='weighted')

        print(f'Test loss: {test_loss:.3f}')
        print(f'Test f1: {f1:.3f}')

        results = {
            'train_history':train_history,
            'val_history':val_history,
            'f1_history':f1_history,
        }

        with open(f'results/{dataset}_{loss}_results.json', 'w') as f:
            json.dump(results, f)
            
        torch.save(model.state_dict(), f'models/{dataset}_{loss}.pth')