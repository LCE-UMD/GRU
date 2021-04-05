'''
model1: recurrent neural network (GRU, low dim last layer)
model2: GRU autoencoder
task: model 1 predicts clip (15 way classifier)
      model 2 reconstructs original time series from hidden states  
data: all runs used together
input to model: ts at current time point
'''
import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA


from dataloader import _clip_class_df, _clip_class_rest_df, K_RUNS
from gru.dataloader import _get_clip_seq as _get_seq
from gru.models import GRUEncoder, GRUDecoder
from gru.cc_utils import _get_true_class_prob, _gru_acc, _gru_test_acc, _gruenc_test_traj
from sklearn.metrics import mean_squared_error, r2_score
from utils import _info
import argparse
import pickle
import time
import os

import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(gpus[0], True)
    tf.config.experimental.set_memory_growth(gpus[1], True)
    tf.config.experimental.set_visible_devices(gpus[1], 'GPU')
except:
    # Invalid device or cannot modify virtual devices once initialized.
    pass

# results directory
RES_DIR = 'results/clip_gru_recon'
if not os.path.exists(RES_DIR):
    os.makedirs(RES_DIR)

K_SEED = 330

def _get_decoder_mask(X_len, max_length, k_feat):    
    
    mask = np.zeros((X_len.numpy().shape[0], max_length, k_feat))
    for ii, length in enumerate(X_len):
        for jj in range(k_feat):
            mask[ii, :length, jj] = 1
        
    return mask

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

def _test(df,args,params):
    '''
    test subject results
    view only for best cross-val parameters
    '''
    _info('test mode')
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
    print('seq lengths = %s' %clip_time)
            
    '''
    init model
    '''
    # get train, test sequences
    X_train, train_len, y_train = _get_seq(df, 
        train_list, args)
    X_test, test_len, y_test = _get_seq(df, 
        test_list, args)
    max_length = tf.math.reduce_max(train_len).numpy()
    
    '''
    train encoder
    '''
    then = time.time()
    model_encoder = GRUEncoder(X_train, args.gru_model_path,
                              k_layers=params['k_layers'], 
                              k_hidden=params['k_hidden'],
                              k_dim=args.k_dim,
                              k_class = args.k_class)
        
    model_encoder.fit(X_train,y_train,epochs=args.num_epochs,
              validation_split=0.2,
              batch_size=args.batch_size,
              verbose=1)
    
    '''
    encoder results
    '''
    results = {}

    a, a_t, c_mtx = _gru_test_acc(model_encoder, X_train, y_train,
                                  clip_time, len(train_list))
    results['train'] = a
    a, a_t, c_mtx = _gru_test_acc(model_encoder, X_test, y_test,
                                  clip_time, len(test_list))
    results['test'] = a
    
    
    '''
    get encoder trajectories
    '''
    traj_train = _gruenc_test_traj(model_encoder, X_train)
    traj_test = _gruenc_test_traj(model_encoder, X_test)
    
    '''
    apply mask on trajectories
    '''
    mask = X_train[:, :, 0] == 0.0
    traj_train[mask, :] = 0.0
    mask = X_test[:, :, 0] == 0.0
    traj_test[mask, :] = 0.0
    
    '''
    train decoder
    '''
    model_decoder = GRUDecoder(traj_train, X_train,k_layers=args.k_layers[0], lr=0.001)

    model_decoder.fit(traj_train, X_train, epochs=args.num_epochs,
                     validation_split=0.2,
                     batch_size=args.batch_size,
                     verbose=1)
    
    '''
    evaluate decoder
    '''
    train_mask = X_train != 0
    test_mask = X_test != 0
    '''
    results on train data
    '''
    outputs = model_decoder.predict(traj_train)
    o = outputs[train_mask==True]
    y = X_train[train_mask==True]
    a = mean_squared_error(o, y)
    print('train_recon mse = %0.3f' %a)
    results['train_mse'] = a
    a = r2_score(o, y)
    results['train_r2'] = a
    print('train_recon r2 = %0.3f' %a)
    '''
    results on test data
    '''
    outputs = model_decoder.predict(traj_test)
    o = outputs[test_mask==True]
    y = X_test[test_mask==True]
    a = mean_squared_error(o, y)
    print('test_recon mse = %0.3f' %a)
    results['test_mse'] = a
    a = r2_score(o, y)
    results['test_r2'] = a
    print('test_recon r2 = %0.3f' %a)
    
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
    # Get all combinations of the parameter grid
    
    _info(args.roi_name)
    # Get all combinations of the parameter grid

    param_grid = {'k_hidden':args.k_hidden,'k_layers':args.k_layers}
    param_grid = [comb for comb in ParameterGrid(param_grid)]

    print(len(param_grid))
    print(len(args.k_layers))

    _info('Number of hyperparameter combinations: '+str(len(param_grid)))
    _info(args.roi_name)
    
    '''
    get dataframe
    '''
    if len(param_grid) == 1:
        res_path = (RES_DIR + 
            '/%s_%d_net_%d' %(args.roi_name, args.roi, args.net) +
            '_trainsize_%d' %(args.train_size) +
            '_k_hidden_%d' %(args.k_hidden[0]) +
            '_kdim_%d' %(args.k_dim) +
            '_k_layers_%d_batch_size_%d' %(args.k_layers[0], args.batch_size) +
            '_num_epochs_%d_z_%d.pkl' %(args.num_epochs, args.zscore))
        
        mod_path = res_path.replace('results','models')
        mod_path = mod_path.replace('pkl','h5')
        
        gru_model_path = mod_path.replace('gru_recon', 'gru')
        gru_model_path = gru_model_path.replace('_kdim_%d' %(args.k_dim), '')
        args.gru_model_path = gru_model_path
        
    elif len(param_grid) > 1:
        res_path = (RES_DIR + 
            '/%s_%d_net_%d' %(args.roi_name, args.roi, args.net) +
            '_trainsize_%d' %(args.train_size) +
            '_kfold_%d' %(args.k_fold) +
            '_batch_size_%d' %(args.batch_size) +
            '_num_epochs_%d_z_%d_GSCV.pkl' %(args.num_epochs, args.zscore))

    if not os.path.isfile(res_path):
        start = time.clock()
#         df = _clip_class_df(args)
        with open('data/df.pkl', 'rb') as f:
            df = pickle.load(f)
        print('data loading time: %.2f seconds' %(time.clock()-start))
        
        if len(param_grid) == 1:
            results = _test(df,args,param_grid[0])
            with open(res_path, 'wb') as f:
                pickle.dump(results, f)
                
        elif len(param_grid) > 1:
            results = {}
            for mm, params in enumerate(param_grid):
                print('---')
                print('model{:02d}'.format(mm) + ': ')
                print(params)
                print('---')
                results['model{:02d}'.format(mm)] = _train(df, args, params)
            # save grid-search CV results
            with open(res_path, 'wb') as f:
                pickle.dump([results, param_grid], f)
            
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='')
    
    # data parameters
    parser.add_argument('-d', '--input-data', type=str,
        default='data/roi_ts', help='path/to/roi/ts')
    parser.add_argument('-r', '--roi', type=int,
        default=300, help='number of ROI')
    parser.add_argument('-n', '--net', type=int,
        default=7, help='number of networks (7 or 17)')
    parser.add_argument('-rn','--roi-name',type=str,
                        default='roi',help='Name of the ROI')
    parser.add_argument('--subnet', type=str,
        default='wb', help='name of subnetwork')
    
    # preprocessing
    parser.add_argument('--zscore', type=int,
        default=1, help='zscore = 1 or 0')

    # training parameters
    parser.add_argument('-k', '--k_fold', type=int,
        default=5, help='number of folds for cross validation')
    parser.add_argument('--k_hidden',nargs='+', type=int,
        default=32, help='size of hidden state (default: 32)')
    parser.add_argument('--k_dim', type=int,
        default=3, help='number of dimensions in low-dim projection')
    parser.add_argument('--k_layers', nargs='+', type=int,
        default=1, help='number of gru layers default: 1')
    parser.add_argument('--batch_size', type=int,
        default=32, help='batch size for training')
    parser.add_argument('--num_epochs', type=int,
        default=45, help='no. of epochs for training')
    parser.add_argument('--train_size', type=int,
        default=100, help='number of participants in training data')
    
    
    args = parser.parse_args()
    
    run(args)