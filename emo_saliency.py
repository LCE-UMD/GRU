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

from gru.dataloader import _emo_class_df
from gru.dataloader import _get_emo_seq as _get_seq
from gru.models import GRUClassifier
from gru.cc_utils import _get_true_class_prob, _gru_acc, _gru_test_acc, _compute_saliency_maps
from utils import _info
import argparse
import pickle
import time
import os

from tqdm import tqdm

import tensorflow as tf
from tensorflow import keras
gpus = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(gpus[0], True)
    tf.config.experimental.set_memory_growth(gpus[1], True)
    tf.config.experimental.set_visible_devices(gpus[1], 'GPU')
except:
    # Invalid device or cannot modify virtual devices once initialized.
    pass

# results directory
RES_DIR = 'results/emo_saliency'
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
    train_list = subject_list[:args.train_size]
    test_list = subject_list[args.train_size:]
    k_class = len(np.unique(df['y']))
    print('number of unique sequences = %d' %k_class)

    features = [ii for ii in df.columns if 'feat' in ii]
    
    gru_model = keras.models.load_model(args.gru_model_path)
    gru_model.trainable = False
    
    mat = {}
    for i_class in range(k_class):
        mat[i_class] = []
    
    for subject in tqdm(test_list):
        subj_df = df[df['Subject']==subject]
        subj_df.reset_index(inplace=True)
        trials = np.split(subj_df,subj_df[subj_df['timepoint']==0].index)[1:]
        for ii,trial in enumerate(trials):
            seq = trial[trial['Subject'] == subject][features].values
            label_seq = trial[trial['Subject']==subject]['y'].values
            class_label = label_seq[0]
            X = [seq]
            X_padded = tf.keras.preprocessing.sequence.pad_sequences(
                X, padding="post", dtype='float')
            
            gradX = _compute_saliency_maps(gru_model, X_padded, i_class)
            
            mat[class_label].append(gradX)
    
    for i_class in range(k_class):
        mat[i_class] = np.stack(mat[i_class], axis=0) #participant x time x vox
    
    return mat


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
            '/%s_near_miss_%d_trainsize_%d' %(args.roi_name,args.near_miss,args.train_size) +
            '_kfold_%d_k_hidden_%d' %(args.k_fold, args.k_hidden[0]) +
            '_k_layers_%d_batch_size_%d' %(args.k_layers[0], args.batch_size) +
            '_num_epochs_%d_z_%d.pkl' %(args.num_epochs, args.zscore))
        
        gru_mod_path = res_path.replace('results','models')
        gru_mod_path = gru_mod_path.replace('pkl','h5')
        gru_model_path = gru_mod_path.replace('saliency', 'gru')
        args.gru_model_path = gru_model_path
        
    elif len(param_grid) > 1:
        res_path = (RES_DIR + 
            '/%s_near_miss_%d_trainsize_%d' %(args.roi_name,args.near_miss,args.train_size) +
            '_kfold_%d_k_hidden_%d' %(args.k_fold, args.k_hidden[0]) +
            '_k_layers_%d_batch_size_%d' %(args.k_layers[0], args.batch_size) +
            '_num_epochs_%d_z_%d.pkl' %(args.num_epochs, args.zscore))

    if not os.path.isfile(res_path):
        df = _emo_class_df(args)
        
        if len(param_grid) == 1:
            mat = _saliency(df, args)
            
            # save results
            with open(res_path, 'wb') as f:
                pickle.dump(mat, f)
                
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
        default='./data/processed/03a-segments_normWithinSubjRun_rAI.pkl',
                        help='path/to/emoprox/rAI/voxelwise/ts')
    parser.add_argument('-r', '--roi_name', type=str,
        default='rvAI', help='number of ROI')
    parser.add_argument('--near_miss', type=int, default = 1,
                        help="near miss dataset? (defualt = 1 (yes))")
    # preprocessing
    parser.add_argument('--zscore', type=int,
        default=0, help='zscore = 1 or 0')

    # training parameters
    parser.add_argument('-k', '--k_fold', type=int,
        default=5, help='number of folds for cross validation')
    parser.add_argument('--k_hidden',nargs='+', type=int,
        default=[32], help='size of hidden state (default: 32)')
    parser.add_argument('--k_layers', nargs='+', type=int,
        default=[1], help='number of gru layers default: 1')
    parser.add_argument('--batch_size', type=int,
        default=32, help='batch size for training')
    parser.add_argument('--num_epochs', type=int,
        default=28, help='no. of epochs for training')
    parser.add_argument('--train_size', type=int,
        default=42, help='number of participants in training data')
    
    
    args = parser.parse_args()
    
    run(args)
    
    print('finished!')