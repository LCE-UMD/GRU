'''
model: GRU
task: predict clip (15 way classifier)
data: all runs used together
input to model: clip time series/seq
output: label time series
'''
import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import KFold

from tensorflow import keras

from dataloader import _clip_class_df, _clip_class_rest_df, K_RUNS
from gru.dataloader import _get_clip_seq as _get_seq
from gru.models import GRUEncoder
from gru.cc_utils import _get_true_class_prob, _gru_acc, _gru_test_acc, _gruenc_test_traj, _compute_saliency_maps
from utils import _info
import argparse
import pickle
import time
import os
import random

from tqdm import tqdm

import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(gpus[0], True)
    tf.config.experimental.set_memory_growth(gpus[1], True)
    tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
except:
    # Invalid device or cannot modify virtual devices once initialized.
    pass

# results directory
RES_DIR = 'results/clip_null_saliency'
if not os.path.exists(RES_DIR):
    os.makedirs(RES_DIR)

K_SEED = 330

def _saliency(df, args):
    '''
    save gradient with respect to the input
    to use as proxy for importance
    '''
    then = time.time()
    _info('save gradients')

    subject_list = np.unique(df['Subject'])
    k_class = len(np.unique(df['y']))
    print('number of unique sequences = %d' %k_class)

    # create columns for gradients
    # don't use number of ROI in case of subnetwork
    features = [ii for ii in df.columns if 'feat' in ii]
    grads = ['grad_%d'%ii for ii in range(len(features))]
    for grad in grads:
        df.loc[:, grad] = np.nan
    
    gru_model = keras.models.load_model(args.gru_model_path)
    gru_model.trainable = False
    
    # length of each clip
    clip_time = np.zeros(k_class, dtype = np.int)
    for i_clip in range(k_class):
        class_df = df[df['y']==i_clip]
        clip_time[i_clip] =int(np.max(np.unique(class_df['timepoint'])) + 1)
    
    num_perms = 1000
    
    for idx_perm in tqdm(range(num_perms)):
        
    
        for i_class in range(k_class):
            
            if i_class != 13:
                continue
            for subject in subject_list:
                
                if i_class==0: # must handle test retest differently
                    seqs = df[(df['Subject']==subject) & 
                        (df['y'] == 0)][features].values
                    gradX = np.zeros(seqs.shape)

                    k_time = int(seqs.shape[0]/K_RUNS)
                    for i_run in range(K_RUNS):
                        seq = seqs[i_run*k_time:(i_run+1)*k_time, :]
                        if args.zscore:
                            # zscore each seq that goes into model
                            seq = (1/np.std(seq))*(seq - np.mean(seq))
                        np.random.shuffle(seq)
                        X = [seq]
                        X_padded = tf.keras.preprocessing.sequence.pad_sequences(
                            X, padding="post", dtype='float')

                        gX = _compute_saliency_maps(gru_model, X_padded, i_class)
                        gradX[i_run*k_time:(i_run+1)*k_time, :] = gX

                else:
                    seq = df[(df['Subject']==subject) & 
                        (df['y'] == i_class)][features].values
                    if args.zscore:
                        # zscore each seq that goes into model
                        seq = (1/np.std(seq))*(seq - np.mean(seq))
                    np.random.shuffle(seq)
                    X = [seq]
                    X_padded = tf.keras.preprocessing.sequence.pad_sequences(
                        X, padding="post", dtype='float')

                    gradX = _compute_saliency_maps(gru_model, X_padded, i_class)
                    gradX = gradX.squeeze()
                    
                df.loc[(df['Subject']==subject) & 
                    (df['y'] == i_class), grads] = gradX

        sal_df = df[['Subject', 'timepoint', 'y'] + grads]
        
        with open("/home/joyneelm/movie-clip-predictions/results/null_saliency/time_shuffle_brokovich/perm_{:03}.pkl".format(idx_perm), 'wb') as f:
                pickle.dump(sal_df, f)
    
    return


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
            '_k_layers_%d_batch_size_%d' %(args.k_layers[0], args.batch_size) +
            '_num_epochs_%d_z_%d.pkl' %(args.num_epochs, args.zscore))
        
        gru_mod_path = res_path.replace('results','models')
        gru_mod_path = gru_mod_path.replace('pkl','h5')
        gru_model_path = gru_mod_path.replace('null_saliency', 'gru')
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
        #df = _clip_class_df(args)
        with open('data/df.pkl', 'rb') as f:
                df = pickle.load(f)
        print('data loading time: %.2f seconds' %(time.clock()-start))
        
        if len(param_grid) == 1:
            _saliency(df, args)    
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
