'''
utils for clip classification (cc)
'''
import numpy as np
import pandas as pd
import pickle
import random
import torch
from torch.nn import Softmax
from torch import backends

from sklearn.metrics import confusion_matrix
from utils import _to_cpu

K_RUNS = 4 #number of runs for each subject

def _get_clip_labels():
    '''
    assign all clips within runs a label
    use 0 for testretest
    '''
    # where are the clips within the run?
    timing_file = pd.read_csv('data/videoclip_tr_lookup.csv')

    clips = []
    for run in range(K_RUNS):
        run_name = 'MOVIE%d' %(run+1) #MOVIEx_7T_yz
        timing_df = timing_file[timing_file['run'].str.contains(run_name)]  
        timing_df = timing_df.reset_index(drop=True)

        for jj, row in timing_df.iterrows():
            clips.append(row['clip_name'])
            
    clip_y = {}
    jj = 1
    for clip in clips:
        if 'testretest' in clip:
            clip_y[clip] = 0
        else:
            clip_y[clip] = jj
            jj += 1

    return clip_y

def _get_mask(X_len, max_length):
        
    mask = np.zeros((len(X_len), max_length))
    for ii, length in enumerate(X_len):
        mask[ii, :length] = 1
        
    return mask

def _get_t_acc(y_hat, y, k_time):
    '''
    accuracy as f(time)
    '''
    a = np.zeros(k_time)
    for ii in range(k_time):
        y_i = y[ii::k_time]
        y_hat_i = y_hat[ii::k_time]
        correct = [1 for p, q in zip(y_i, y_hat_i) if p==q]
        a[ii] = sum(correct)/len(y_i)
        
    return a

def _get_confusion_matrix(y, predicted):
    '''
    if cuda tensor, must move to cpu first
    '''
    y, p = _to_cpu(y), _to_cpu(predicted)

    return confusion_matrix(y, p)

def _ff_acc(model, X, y, X_len, max_length,
    clip_time, return_states=False):
    '''
    masked accuracy for feedforward network
    '''
    # mask to ignore padding
    mask = _get_mask(X_len, max_length)
    
    # forward pass
    if return_states:
        _, outputs = model(X)
    else:
        outputs = model(X)

    # logits to labels
    _, y_hat = torch.max(outputs, 2)
    
    # remove padded values
    # converts matrix to vec
    y_hat = y_hat[mask==True]
    y = y[mask==True]

    # accuracy for all t
    correct = (y_hat==y).sum().item()
    a = correct/(mask==True).sum()

    # accuracy as a function of t
    k_class = len(clip_time)
    a_t = {}
    for ii in range(k_class):
        y_i = y[y==ii]
        y_hat_i = y_hat[y==ii]
        k_time = clip_time[ii]
        a_t[ii] = _get_t_acc(y_hat_i, y_i, k_time)

    c_mtx = _get_confusion_matrix(y, y_hat)
    
    return a, a_t, c_mtx

def _ff_test_acc(model, X, y, X_len, max_length,
    clip_time, k_sub, return_states=False):
    '''
    masked accuracy for ff
    per participant accuracy
    '''
    # mask to ignore padding
    mask = _get_mask(X_len, max_length)
    
    # forward pass
    if return_states:
        _, outputs = model(X)
    else:
        outputs = model(X)

    # logits to labels
    _, y_hat = torch.max(outputs, 2)
    
    # remove padded values
    # converts matrix to vec
    y_hat = y_hat[mask==True]
    y = y[mask==True]

    a = np.zeros(k_sub)
    sub_size = len(y_hat)//k_sub
    for s in range(k_sub):
        # group based on k_sub
        y_hat_s = y_hat[s*sub_size:(s+1)*sub_size]
        y_s = y[s*sub_size:(s+1)*sub_size]
        # accuracy for each group
        correct = (y_hat_s==y_s).sum().item()
        a[s] = correct/len(y_s)
    
    # accuracy as a function of t
    k_class = len(clip_time)
    a_t = {}
    for ii in range(k_class):
        y_i = y[y==ii]
        y_hat_i = y_hat[y==ii]
        k_time = clip_time[ii]
        a_t[ii] = np.zeros((k_sub, k_time))
        sub_size = len(y_hat_i)//k_sub
        for s in range(k_sub):
            # group based on k_sub
            y_hat_s = y_hat_i[s*sub_size:(s+1)*sub_size]
            y_s = y_i[s*sub_size:(s+1)*sub_size]
            # accuracy for each group
            a_t[ii][s] = _get_t_acc(y_hat_s, y_s, k_time)

    c_mtx = _get_confusion_matrix(y, y_hat)
    
    return a, a_t, c_mtx

def _lstm_acc(model, X, y, X_len, max_length,
    clip_time, return_states=False):
    '''
    masked accuracy for lstm
    '''
    # mask to ignore padding
    mask = _get_mask(X_len, max_length)
    
    # forward pass
    if return_states:
        _, outputs = model(X, X_len, max_length)
    else:
        outputs = model(X, X_len, max_length)

    # logits to labels
    _, y_hat = torch.max(outputs, 2)
    
    # remove padded values
    # converts matrix to vec
    y_hat = y_hat[mask==True]
    y = y[mask==True]

    # accuracy for all t
    correct = (y_hat==y).sum().item()
    a = correct/(mask==True).sum()

    # accuracy as a function of t
    k_class = len(clip_time)
    a_t = {}
    for ii in range(k_class):
        y_i = y[y==ii]
        y_hat_i = y_hat[y==ii]
        k_time = clip_time[ii]
        a_t[ii] = _get_t_acc(y_hat_i, y_i, k_time)

    c_mtx = _get_confusion_matrix(y, y_hat)
    
    return a, a_t, c_mtx

def _lstm_test_acc(model, X, y, X_len, max_length,
    clip_time, k_sub, return_states=False):
    '''
    masked accuracy for lstm
    per participant accuracy
    '''
    # mask to ignore padding
    mask = _get_mask(X_len, max_length)
    
    # forward pass
    if return_states:
        _, outputs = model(X, X_len, max_length)
    else:
        outputs = model(X, X_len, max_length)

    # logits to labels
    _, y_hat = torch.max(outputs, 2)
    
    # remove padded values
    # converts matrix to vec
    y_hat = y_hat[mask==True]
    y = y[mask==True]

    a = np.zeros(k_sub)
    sub_size = len(y_hat)//k_sub
    for s in range(k_sub):
        # group based on k_sub
        y_hat_s = y_hat[s*sub_size:(s+1)*sub_size]
        y_s = y[s*sub_size:(s+1)*sub_size]
        # accuracy for each group
        correct = (y_hat_s==y_s).sum().item()
        a[s] = correct/len(y_s)
    
    # accuracy as a function of t
    k_class = len(clip_time)
    a_t = {}
    for ii in range(k_class):
        y_i = y[y==ii]
        y_hat_i = y_hat[y==ii]
        k_time = clip_time[ii]
        a_t[ii] = np.zeros((k_sub, k_time))
        sub_size = len(y_hat_i)//k_sub
        for s in range(k_sub):
            # group based on k_sub
            y_hat_s = y_hat_i[s*sub_size:(s+1)*sub_size]
            y_s = y_i[s*sub_size:(s+1)*sub_size]
            # accuracy for each group
            a_t[ii][s] = _get_t_acc(y_hat_s, y_s, k_time)

    c_mtx = _get_confusion_matrix(y, y_hat)
    
    return a, a_t, c_mtx

class VanillaSaliency():
    """
    gradients generated with vanilla back propagation
    for sequence data
    adapted from: https://github.com/utkuozbulak/pytorch-cnn-visualizations/blob/master/src/vanilla_backprop.py
    """
    def __init__(self, model, device, multiply_seq=True):
        '''
        multiple_seq: Bool, for grad*Image equivalent
        '''
        self.model = model
        
        # ensure model is in eval mode
        self.model.eval()
        
        self.device = device
        self.multiply_seq = multiply_seq
        
    def generate_gradients(self, input_seq,
        seq_length, label):
        '''
        input_seq: 1 x time x feat
        seq_length: list(time)
        labels for clip classification are of the form [label]*time
        so here, simply input label at first timepoint (label[0])
        '''
        # to bypass "RuntimeError: cudnn RNN backward can only be called in training mode"
        with backends.cudnn.flags(enabled=False):
            
            # forward
            model_output = self.model(input_seq, seq_length)
            
            # zero grads
            self.model.zero_grad()
            
            # target for backprop
            one_hot_output = torch.FloatTensor(1, seq_length[0], 
                model_output.shape[-1]).zero_().to(self.device)
            one_hot_output[0][:, label] = 1
            
            # backward pass
            model_output.backward(gradient=one_hot_output)

            # gradient with respect to input 
            # [0]: squeeze 1 x t x n
            if self.multiply_seq:
                g = input_seq.grad * input_seq
            else:
                g = input_seq.grad

            grad = _to_cpu(g).numpy()[0]
            
        return grad

################################### Chirag Addition ###################################

# Get probability of the true class
def _prob_true_class(model, X, y, X_len, max_length, return_states=False):
    '''
    Probability of belonging to true class
    
    Inputs
    ------
    model: trained model
    X: batch_size x time x voxels
    y: labels (batch_size x time)
    X_len: 1D tensor equal to the length of batch size.
            Every entry is length of the sequence
    max_length: maximum length of a segment.            
            
    Returns
    -------
    results: (dict) probability of belonging to the true class at each timepoint
    '''
    
    # mask to ignore padding
    mask = _get_mask(X_len, max_length)
    
    # forward pass
    if return_states:
        _, outputs = model(X, X_len, max_length)
    else:
        outputs = model(X, X_len, max_length)
    
    # Apply sofmax function to the output scores
    sm = Softmax(2) 
    probs = sm(outputs)
    
    results = {}
    for _class in 'approach retreat'.split():
        results[_class] = {}
        for tp in range(y.shape[1]):
            results[_class][tp] = []
        
    for tp in range(y.shape[1]):
        # Probability of belonging to retreat (0)
        y_masked = y[mask[:,tp]==1,tp]
        probs_masked = probs[mask[:,tp]==1,tp,:]
    
        y_retreat_mask = y_masked == 0
        y_approach_mask = y_masked == 1

        results['approach'][tp].extend(probs_masked[y_approach_mask,1].cpu().detach().numpy().tolist())
        results['retreat'][tp].extend(probs_masked[y_retreat_mask,0].cpu().detach().numpy().tolist())
        
    return results
        
    