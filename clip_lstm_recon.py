'''
model1: recurrent neural network (lstm, low dim last layer)
model2: lstm autoencoder
task: model 1 predicts clip (15 way classifier)
      model 2 reconstructs original time series from hidden states  
data: all runs used together
input to model: ts at current time point
'''
import numpy as np
import pandas as pd
import pickle
import os
import argparse
import time
'''
ml
'''
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from models import LSTMEncoder, LSTMDecoder
'''
Helpers
'''
from utils import _info, _to_cpu
from cc_utils import _lstm_acc, _lstm_test_acc
from dataloader import _get_clip_seq as _get_seq
from dataloader import _clip_class_df

# results directory
RES_DIR = 'results/clip_lstm_recon'
if not os.path.exists(RES_DIR):
    os.makedirs(RES_DIR)

K_SEED = 330
K_RUNS = 4

def _get_decoder_seq(df, subject_list, model, args):
    '''
    return:
    X: input seq (batch_size x time x hidden_state_size)
    y: output seq (batch_size x time x feat_size)
    X_len: len of each seq (batch_size x 1)
    time <-> max length after padding
    '''
    features = [ii for ii in df.columns if 'feat' in ii]
    # forward pass for encoder model
    model.eval()

    X = []
    y = []
    for subject in subject_list:
        for i_class in range(args.k_class):
            
            if i_class==0: # split test-retest into 4
                seqs = df[(df['Subject']==subject) & 
                          (df['y'] == 0)][features].values
                label_seqs = df[(df['Subject']==subject) & 
                                (df['y'] == 0)]['y'].values

                k_time = int(seqs.shape[0]/K_RUNS)
                for i_run in range(K_RUNS):
                    seq = seqs[i_run*k_time:(i_run+1)*k_time, :]
                    label_seq = label_seqs[i_run*k_time:(i_run+1)*k_time]
                    if args.zscore:
                        # zscore each seq that goes into model
                        seq = (1/np.std(seq))*(seq - np.mean(seq))

                    S = [torch.FloatTensor(seq)]
                    lenS = torch.LongTensor([len(seq)]).to(args.device)

                    # pad sequences
                    S = pad_sequence(S, batch_first=True, 
                                     padding_value=0).to(args.device)
                    
                    # forward pass
                    hidden_seq, _ = model(S, lenS) 
                    hidden_seq = _to_cpu(hidden_seq).numpy()[0] # t x k array
                    X.append(torch.FloatTensor(hidden_seq))
                    y.append(torch.FloatTensor(seq))
            else:
                seq = df[(df['Subject']==subject) & 
                         (df['y'] == i_class)][features].values
                label_seq = df[(df['Subject']==subject) & 
                               (df['y'] == i_class)]['y'].values
                if args.zscore:
                    # zscore each seq that goes into model
                    seq = (1/np.std(seq))*(seq - np.mean(seq))
                
                S = [torch.FloatTensor(seq)]
                lenS = torch.LongTensor([len(seq)]).to(args.device)

                # pad sequences
                S = pad_sequence(S, batch_first=True,
                                 padding_value=0).to(args.device)
                
                # forward pass
                hidden_seq, _ = model(S, lenS) 
                hidden_seq = _to_cpu(hidden_seq).numpy()[0] # t x k array
                X.append(torch.FloatTensor(hidden_seq))
                y.append(torch.FloatTensor(seq))
            
    X_len = torch.LongTensor([len(seq) for seq in X])

    # pad sequences
    X = pad_sequence(X, batch_first=True, padding_value=0)
    y = pad_sequence(y, batch_first=True, padding_value=0)
            
    return X.to(args.device), X_len.to(args.device), y.to(args.device)

def _zscore_seq(df, args):
    '''
    helper to zscore each sequence for PCA
    '''
    subject_list = np.unique(df['Subject'])
    features = [ii for ii in df.columns if 'feat' in ii]

    X = np.empty((0, len(features)))
    for subject in subject_list:
        for i_class in range(args.k_class):
            
            if i_class==0: #handle test retest differently
                seqs = df[(df['Subject']==subject) & 
                          (df['y'] == 0)][features].values
                k_time = int(seqs.shape[0]/K_RUNS)
                for i_run in range(K_RUNS):
                    seq = seqs[i_run*k_time:(i_run+1)*k_time, :]
                    # zscore each seq that goes into model
                    seq = (1/np.std(seq))*(seq - np.mean(seq)) # t x feat_size
                    X = np.vstack((X, seq))

            else:
                seq = df[(df['Subject']==subject) & 
                         (df['y'] == i_class)][features].values
                # zscore each seq that goes into model
                seq = (1/np.std(seq))*(seq - np.mean(seq)) # t x feat_size
                X = np.vstack((X, seq))

    return X

def _get_pca_recon(df, train_list, test_list, args):
    
    features = [ii for ii in df.columns if 'feat' in ii]
    
    train_df = df[df['Subject'].isin(train_list)]
    if args.zscore:
        X_train = _zscore_seq(train_df, args)
    else:
        X_train = train_df[features].values

    test_df = df[df['Subject'].isin(test_list)]
    if args.zscore:
        X_test = _zscore_seq(test_df, args)
    else:
        X_test = test_df[features].values
    
    then = time.time()
    #
    pca = PCA(n_components=args.k_dim)
    X_pc_train = pca.fit_transform(X_train)
    X_pc_test = pca.transform(X_test)
    print('--- pca transform time =  {:0.4f} seconds ---'.format(
        time.time() - then))
    
    pc_var = pca.explained_variance_ratio_
    # inverse transform for reconstruction
    X_hat_train = pca.inverse_transform(X_pc_train)
    X_hat_test = pca.inverse_transform(X_pc_test)

    train_mse = mean_squared_error(X_train, X_hat_train)
    test_mse = mean_squared_error(X_test, X_hat_test)

    train_r2 = r2_score(X_train, X_hat_train)
    test_r2 = r2_score(X_test, X_hat_test)

    return train_mse, test_mse, train_r2, test_r2, pc_var

def _get_decoder_mask(X_len, max_length, k_feat):
        
    mask = torch.zeros(len(X_len), max_length, k_feat)
    for ii, length in enumerate(X_len):
        for jj in range(k_feat):
            mask[ii, :length, jj] = 1
        
    return mask

def _test(df, args):
    
    _info('test mode')
    
    # set pytorch device
    torch.manual_seed(K_SEED)
    use_cuda = torch.cuda.is_available()
    args.device = torch.device('cuda:0' if use_cuda else 'cpu')
    if use_cuda:
        _info('cuda')
    else:
        _info('cpu')

    # get X-y from df
    subject_list = df['Subject'].unique()
    train_list = subject_list[:args.train_size]
    test_list = subject_list[args.train_size:]

    print('number of subjects = %d' %(len(subject_list)))
    features = [ii for ii in df.columns if 'feat' in ii]
    k_feat = len(features)
    print('number of features = %d' %(k_feat))
    args.k_class = len(np.unique(df['y']))
    print('number of classes = %d' %(args.k_class))
    
    # length of each clip
    clip_time = np.zeros(args.k_class)
    for ii in range(args.k_class):
        class_df = df[df['y']==ii]
        clip_time[ii] = np.max(np.unique(class_df['timepoint'])) + 1
    clip_time = clip_time.astype(int) # df saves float
    _info('seq lengths = %s' %clip_time)
        
    # results dict init
    results = {}    
    
    '''
    encoder main
    '''
    model_enc = LSTMEncoder(k_feat, args.k_hidden, args.k_dim,
                            args.k_layers, args.k_class,
                            return_states = True)
    model_enc.to(args.device)
    print(model_enc)
    
    lossfn = nn.CrossEntropyLoss(ignore_index=-100) 
    # if input is cuda, loss function is auto cuda
    opt = torch.optim.Adam(model_enc.parameters())

    X_train, train_len, y_train = _get_seq(
        df, train_list, args)
    X_test, test_len, y_test = _get_seq(
        df, test_list, args)
    max_length = torch.max(train_len)
    '''
    train encoder
    '''
    permutation = torch.randperm(X_train.size()[0])
    losses = np.zeros(args.num_epochs)

    #
    then = time.time()

    for epoch in range(args.num_epochs):
        for i in range(0, X_train.size()[0], args.batch_size):
            
            indices = permutation[i:i + args.batch_size]
            batch_x, batch_y = X_train[indices], y_train[indices]
            batch_x_len = train_len[indices]
            
            _, y_pred = model_enc(batch_x, batch_x_len, max_length)
            loss = lossfn(y_pred.view(-1, args.k_class), 
                          batch_y.view(-1))
            
            opt.zero_grad()
            loss.backward()
            opt.step()
        
        losses[epoch] = loss
    
    _info(losses)
    #
    print('--- enc train time =  {:0.4f} seconds ---'.format(
        time.time() - then))
    
    '''
    results on train data
    '''
    a, a_t, c_mtx = _lstm_test_acc(
        model_enc, X_train, y_train, train_len, max_length, 
        clip_time, len(train_list), return_states = True)
    results['train'] = a
    print('tacc = %0.3f' %np.mean(a))
    '''
    results on test data
    '''
    a, a_t, c_mtx = _lstm_test_acc(
        model_enc, X_test, y_test, test_len, max_length, 
        clip_time, len(test_list), return_states = True)
    results['test'] = a
    print('sacc = %0.3f' %np.mean(a))
    
    '''
    decoder main
        input: 
            k_input <- args.k_dim
            k_hidden <- k_feat
    '''
    # input to decoder:
    model_dec = LSTMDecoder(args.k_dim, k_feat, args.k_layers)
    model_dec.to(args.device)
    print(model_dec)
    
    lossfn = nn.MSELoss()
    opt = torch.optim.Adam(model_dec.parameters())

    X_train, train_len, y_train = _get_decoder_seq(
        df, train_list, model_enc, args)
    X_test, test_len, y_test = _get_decoder_seq(
        df, test_list, model_enc, args)
    max_length = torch.max(train_len)

    del model_enc # free up gpu
    if use_cuda:
        torch.cuda.empty_cache()
    '''
    train decoder
    '''
    permutation = torch.randperm(X_train.size()[0])
    losses = np.zeros(args.num_epochs)

    #
    then = time.time()
    for epoch in range(args.num_epochs):
        for i in range(0, X_train.size()[0], args.batch_size):
            
            indices = permutation[i:i + args.batch_size]
            batch_x, batch_y = X_train[indices], y_train[indices]
            batch_x_len = train_len[indices]
            batch_mask = _get_decoder_mask(
                batch_x_len, max_length, k_feat)
            
            y_pred = model_dec(batch_x, batch_x_len, max_length)
            loss = lossfn(y_pred[batch_mask==True],
                          batch_y[batch_mask==True])
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            
        losses[epoch] = loss

    _info(losses)
    #
    print('--- dec train time =  {:0.4f} seconds ---'.format(
        time.time() - then))
    '''
    evaluate decoder
    '''
    model_dec.eval()

    # masks are on cpu
    train_mask = _get_decoder_mask(train_len, max_length, k_feat)
    test_mask = _get_decoder_mask(test_len, max_length, k_feat)
    '''
    results on train data
    '''
    outputs = model_dec(X_train, train_len, max_length)
    o = _to_cpu(outputs[train_mask==True])
    y = _to_cpu(y_train[train_mask==True])
    a = mean_squared_error(o, y)
    results['train_mse'] = a
    a = r2_score(o, y)
    results['train_r2'] = a
    print('t_recon r2 = %0.3f' %a)
    '''
    results on test data
    '''
    outputs = model_dec(X_test, test_len, max_length)
    o = _to_cpu(outputs[test_mask==True])
    y = _to_cpu(y_test[test_mask==True])
    a = mean_squared_error(o, y)
    results['test_mse'] = a
    a = r2_score(o, y)
    results['test_r2'] = a
    print('s_recon r2 = %0.3f' %a)

    del model_dec # free up gpu
    if use_cuda:
        torch.cuda.empty_cache()
    '''
    compare to pca reconstruction
    '''
    train_mse, test_mse, train_r2, test_r2, pca_var = _get_pca_recon(
        df, train_list, test_list, args)
    results['pca_var']= pca_var
    '''
    results on train data
    '''
    results['train_pca_mse'] = train_mse
    results['train_pca_r2'] = train_r2
    print('t_pca_recon r2 = %0.3f' %train_r2)
    '''
    results on test data
    '''
    results['test_pca_mse'] = test_mse
    results['test_pca_r2'] = test_r2
    print('s_pca_recon r2 = %0.3f' %test_r2)
    
    return results

def run(args):   
    
    _info(args.subnet)
    '''
    get dataframe
    '''
    # all-but-subnetwork (invert_flag)
    if 'minus' in args.subnet:
        args.invert_flag = True

    res_path = (RES_DIR + 
        '/roi_%d_net_%d' %(args.roi, args.net) + 
        '_nw_%s' %(args.subnet) +
        '_trainsize_%d' %(args.train_size) +
        '_kfold_%d_k_hidden_%d' %(args.k_fold, args.k_hidden) +
        '_kdim_%d' %(args.k_dim) +
        '_k_layers_%d_batch_size_%d' %(args.k_layers, args.batch_size) +
        '_num_epochs_%d_z_%d.pkl' %(args.num_epochs, args.zscore))
    if not os.path.isfile(res_path):
        df = _clip_class_df(args)
        results = _test(df, args)
        with open(res_path, 'wb') as f:
            pickle.dump(results, f)
            
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='')
    
    # data parameters
    parser.add_argument('-d', '--input-data', type=str,
        default='data/roi_ts', help='path/to/roi/data')
    parser.add_argument('-r', '--roi', type=int,
        default=300, help='number of ROI')
    parser.add_argument('-n', '--net', type=int,
        default=7, help='number of networks (7 or 17)')
    parser.add_argument('--subnet', type=str,
        default='wb', help='name of subnetwork')
    
    # preprocessing
    parser.add_argument('--zscore', type=int,
        default=1, help='zscore = 1 or 0')

    # training parameters
    parser.add_argument('-k', '--k_fold', type=int,
        default=10, help='number of folds for cross validation')
    parser.add_argument('--k_hidden', type=int,
        default=150, help='size of hidden state')
    parser.add_argument('--k_dim', type=int,
        default=3, help='number of dimensions in low-dim projection')
    parser.add_argument('--k_layers', type=int,
        default=1, help='number of lstm layers')
    parser.add_argument('--batch_size', type=int,
        default=16, help='batch size for training')
    parser.add_argument('--num_epochs', type=int,
        default=50, help='no. of epochs for training')
    parser.add_argument('--train_size', type=int,
        default=100, help='number of participants in training data')
    
    args = parser.parse_args()
    
    run(args)

    print('finished!')
