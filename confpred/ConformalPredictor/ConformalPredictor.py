import numpy as np
from abc import ABC, abstractmethod
from entmax.activations import entmax15, sparsemax
import torch
class ConformalPredictor():
    def __init__(self,score):
        self.score = score
    
    def calibrate(self, cal_true, cal_pred, alpha):
        n_cal = cal_true.shape[0]
        cal_scores = self.score.get_single_score(cal_true,cal_pred)
        q_level = np.ceil((n_cal+1)*(1-alpha))/n_cal
        try:
            self.q_hat = np.quantile(cal_scores, q_level, method = 'higher')
        except TypeError:
            self.q_hat = np.quantile(cal_scores, q_level, interpolation = 'higher')
    
    def predict(self, test_pred, disallow_empty = False,use_temperature = False):
        
        if use_temperature:
            if self.score.alpha == 1.5:
                qhat = self.q_hat
                pred_ent = entmax15((2/qhat)*torch.tensor(test_pred), dim=-1).numpy()
                test_match = pred_ent>0
            elif self.score.alpha == 2:
                qhat = self.q_hat
                pred_ent = sparsemax((1/qhat)*torch.tensor(test_pred), dim=-1).numpy()
                test_match = pred_ent>0
            else:
                raise ValueError('Temperature only supported for alpha = 1.5 or 2')
        else:
            test_scores = self.score.get_multiple_scores(test_pred)
            test_match = test_scores<= self.q_hat

        if disallow_empty:
            helper = np.zeros(test_pred[(test_match.sum(axis = 1)==0)].shape)
            helper[np.arange(helper.shape[0]),test_pred[(test_match.sum(axis = 1)==0)].argmax(axis = 1)]=1
            test_match[(test_match.sum(axis = 1)==0)] = helper
        
        return test_match
    
    def evaluate(self, test_true, test_pred, disallow_empty = False,use_temperature = False):
        n_test = test_pred.shape[0]
        test_match = self.predict(test_pred, disallow_empty,use_temperature)
        set_size = test_match.sum(axis = 1).mean()
        coverage = test_match[test_true.astype(bool)].sum()/n_test
        return set_size, coverage

class ConformalScore(ABC):
    
    @abstractmethod
    def get_single_score(self, cal_true, cal_pred) -> np.array:
        pass
    
    @abstractmethod
    def get_multiple_scores(self, test_pred) -> np.array:
        pass
    
class SparseScore(ConformalScore):
    def __init__(self, alpha):
        self.alpha = alpha
        
    def get_single_score(self, cal_true, cal_pred) -> np.array:
        ranks = np.flip(cal_pred.argsort(axis = 1),axis = 1).argsort()
        match = np.select(cal_true.astype(bool).T,ranks.T)
        cond = ranks>np.expand_dims(match, axis=-1)
        k_y = np.select(cal_true.astype(bool).T,cal_pred.T)
        output = (cal_pred-np.expand_dims(k_y, axis=-1))
        output[cond] = 0
        return np.linalg.norm(output,axis = 1, ord = 1/(self.alpha-1))
    
    def get_multiple_scores(self, test_pred) -> np.array:
        output = []
        for i in range(test_pred.shape[1]):
            true_test = np.zeros(test_pred.shape)
            true_test[:,i] = 1
            output.append(self.get_single_score(true_test,test_pred)[None,:])
        return np.concatenate(output,axis=0).T
    
class SoftmaxScore(ConformalScore):
    def get_single_score(self, cal_true, cal_pred) -> np.array:
        
        true_mask = cal_true.astype(bool)
        cal_scores = 1 - cal_pred[true_mask]
        return cal_scores
    
    def get_multiple_scores(self, test_pred) -> np.array:
        
        return 1 - test_pred