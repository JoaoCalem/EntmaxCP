from confpred.classifier import CNN, evaluate
from confpred.datasets import CIFAR10, CIFAR100, MNIST

from sklearn.metrics import f1_score,accuracy_score
import torch
import numpy as np
import pickle
from entmax.losses import SparsemaxLoss

loss = 'softmax' #sparsemax, softmax or entmax15
transformation = 'logits'
dataset='CIFAR100' #CIFAR100 or MNIST

device = 'cuda:1' if torch.cuda.is_available() and torch.cuda.device_count()>1 else 'cuda' if torch.cuda.is_available() else 'cpu'

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

model.load_state_dict(torch.load(f'models/{dataset}_{loss}.pth'))
if loss == 'sparsemax':
    criterion = SparsemaxLoss()
elif loss == 'softmax':
    criterion = torch.nn.NLLLoss()
test_proba, test_pred, test_true, test_loss = evaluate(
                                                    model,
                                                    data.test,
                                                    criterion,
                                                    True)

test_f1 = f1_score(test_true, test_pred, average='weighted')
test_acc = accuracy_score(test_true, test_pred)

print(f'Test loss: {test_loss:.3f}')
print(f'Test f1: {test_f1:.3f}')
print(f'Test Accuracy: {test_acc:.3f}')

cal_proba, cal_pred, cal_true, cal_loss = evaluate(
                                                    model,
                                                    data.cal,
                                                    criterion,
                                                    True)

cal_f1 = f1_score(cal_true, cal_pred, average='weighted')
cal_acc = accuracy_score(cal_true, cal_pred)

print(f'Calibration loss: {cal_loss:.3f}')
print(f'Calibration f1: {cal_f1:.3f}')
print(f'Calibration acc: {cal_acc:.3f}')

#One Hot Encoding
test_true_enc = np.zeros((test_true.size, test_true.max()+1), dtype=int)
test_true_enc[np.arange(test_true.size),test_true] = 1

cal_true_enc = np.zeros((cal_true.size, cal_true.max()+1), dtype=int)
cal_true_enc[np.arange(cal_true.size),cal_true] = 1



predictions = {'test':{'proba':test_proba,'true':test_true_enc},
 'cal':{'proba':cal_proba,'true':cal_true_enc}}

loss = "NLLLoss" if loss=="softmax" else "FYLoss"
for dataset_type in ['cal','test']:
    with open(f'predictions/{dataset}_{dataset_type}_true.pickle', 'wb') as f:
        pickle.dump(predictions[dataset_type]['true'], f)
    with open(
        f'predictions/{dataset}_{dataset_type}_{loss}' +
            f'_{transformation}_{"proba"}.pickle'
        , 'wb'
    ) as f:
        pickle.dump(predictions[dataset_type]["proba"], f)