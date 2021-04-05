'''
model: gru
task: predict behavioral score (at every t***: predict for each hidden state)
data: all clips used together
behavioral measures: see notebook
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
from sklearn.model_selection import KFold
import tensorflow as tf

from gru.models import GRURegressor
model_type = 'gru'
'''
Helpers
'''
from utils import _info
from gru.rb_utils import dnn_score, mse, Rsquared
from dataloader import _bhv_class_df as _bhv_reg_df
from gru.dataloader import _get_bhv_seq as _get_seq

# results directory
RES_DIR = 'results/bhv_{}'.format(model_type)
if not os.path.exists(RES_DIR):
    os.makedirs(RES_DIR)
    
K_SEED = 330

'''
SCORES:
'mse': mean squared error
'p': pearson correlation
's': spearman correlation
'''
SCORES = ['mse', 'p', 's']

def _train(df, bhv_df, args):
    # get X-y from df
    feature = [ii for ii in df.columns if 'feat' in ii]
    k_feat = len(feature)
    print('number of features = %d' %(k_feat))
    k_clip = len(np.unique(df['c']))
    print('number of clip = %d' %(k_clip))
    subject_list = bhv_df['Subject'].unique()
    train_list = subject_list[:args.train_size]
    test_list = subject_list[args.train_size:]

    # length of each clip
    clip_time = np.zeros(k_clip)
    for ii in range(k_clip):
        class_df = df[df['c']==ii]
        clip_time[ii] = np.max(np.unique(class_df['timepoint'])) + 1
    clip_time  = clip_time.astype(int) # df saves float
    _info('seq lengths = %s' %clip_time)

    # init dict for all results
    results = {}

    # true and predicted scores and clip label
    results['y'] = {}
    results['y_hat'] = {}
    results['c'] = {}

    for score in SCORES:
        # mean scores across time
        results['train_%s'%score] = np.zeros(args.k_fold)
        results['val_%s'%score] = np.zeros(args.k_fold)

        # per clip temporal score
        results['t_train_%s'%score] = {}
        results['t_val_%s'%score] = {}

        for ii in range(k_clip):
            results['t_train_%s'%score][ii] = np.zeros(
                (args.k_fold, clip_time[ii]))
            results['t_val_%s'%score][ii] = np.zeros(
                (args.k_fold, clip_time[ii]))

    kf = KFold(n_splits=args.k_fold, random_state=K_SEED)

    # get participant lists for each assigned class
    # ensure they're only in train_list
    class_list = {}
    for ii in range(args.k_class):
        class_list[ii] = bhv_df[
            (bhv_df['Subject'].isin(train_list)) &
            (bhv_df['y']==ii)]['Subject'].values
        print('No. of participants in class {} = {}'.format(
            ii, len(class_list[ii])))
    '''    
    split participants in each class with kf
    nearly identical ratio of train and val,
    in all classes
    '''
    split = {}
    for ii in range(args.k_class):
        split[ii] = kf.split(class_list[ii])

    for i_fold in range(args.k_fold):

        _info('fold: %d/%d' %(i_fold+1, args.k_fold))

        # ***between-subject train-val split
        train_subs, val_subs = [],[]
        for ii in range(args.k_class):
            train, val = next(split[ii])
            for jj in train:
                train_subs.append(class_list[ii][jj])
            for jj in val:
                val_subs.append(class_list[ii][jj])
        '''
        model main
        '''

        X_train, train_len, y_train, c_train = _get_seq(
            df,train_subs,args)
        X_val, val_len, y_val, c_val = _get_seq(
            df, val_subs, args)

        max_length = tf.reduce_max(train_len)

        '''
        train regression model
        '''
        then = time.time()
        model = GRURegressor(X_train, k_hidden=args.k_hidden, 
                             k_layers=args.k_layers,
                             l2=args.l2,dropout=args.dropout,lr=args.lr)
        model.fit(X_train,y_train.reshape(y_train.shape[0],y_train.shape[1],1),
                  batch_size=args.batch_size,
                  epochs=args.num_epochs,verbose=1,
                  validation_split=0.2)

        print('--- train time =  %0.4f seconds ---' %(time.time() - then))

        '''
        results on train data
        '''
        s, s_t, _, _, _ = dnn_score(
            model, X_train, y_train,
            c_train, train_len, max_length, 
            clip_time, model_type = model_type)
        for score in SCORES:
            results['train_%s'%score][i_fold] = s[score]
            for ii in range(k_clip):
                results['t_train_%s'%score][ii][i_fold] = s_t[ii][score]
        print('train p = %0.3f' %s['p'])
        '''
        results on val data
        '''
        s, s_t, y, y_hat, c = dnn_score(
            model, X_val, y_val,
            c_val, val_len, max_length, 
            clip_time, model_type = model_type)
        for score in SCORES:
            results['val_%s'%score][i_fold] = s[score]
            for ii in range(k_clip):
                results['t_val_%s'%score][ii][i_fold] = s_t[ii][score]
        print('val p = %0.3f' %s['p'])

        results['y'][i_fold] = y
        results['y_hat'][i_fold] = y_hat
        results['c'][i_fold] = c

    return results

def _test(df, bhv_df, args):

    _info('test mode')

    # get X-y from df   
    features = [ii for ii in df.columns if 'feat' in ii]
    k_feat = len(features)
    print('number of features = %d' %(k_feat))
    k_clip = len(np.unique(df['c']))
    print('number of clips = %d' %(k_clip))
    subject_list = bhv_df['Subject'].unique()
    train_list = subject_list[:args.train_size]
    test_list = subject_list[args.train_size:]

    # length of each clip
    clip_time = np.zeros(k_clip)
    for ii in range(k_clip):
        class_df = df[df['c']==ii]
        clip_time[ii] = np.max(np.unique(class_df['timepoint'])) + 1
    clip_time = clip_time.astype(int) # df saves float
    _info('seq lengths = %s' %clip_time)

    # init dict for all results
    results = {}
    for score in SCORES:

        # per clip temporal score
        results['t_train_%s'%score] = {}
        results['t_test_%s'%score] = {}

        for ii in range(k_clip):
            results['t_train_%s'%score][ii] = np.zeros(clip_time[ii])
            results['t_test_%s'%score][ii] = np.zeros(clip_time[ii])
    '''
    model main
    '''

    # get train, test sequences
    X_train, train_len, y_train, c_train = _get_seq(
        df, train_list, args)
    X_test, test_len, y_test, c_test = _get_seq(
        df, test_list, args)

    max_length = tf.reduce_max(train_len)

    '''
    test regression model
    '''
    then = time.time()
    model = GRURegressor(X_train, k_hidden=args.k_hidden, 
                                 k_layers=args.k_layers,
                                 l2=args.l2,dropout=args.dropout,lr=args.lr)
    model.fit(X_train,y_train.reshape(y_train.shape[0],y_train.shape[1],1),
              batch_size=args.batch_size,
              epochs=args.num_epochs,verbose=1,
              validation_split=0.2)
    print('--- train time =  %0.4f seconds ---' %(time.time() - then))

    '''
    results on train data
    '''
    s, s_t, _, _, _ = dnn_score(
        model, X_train, y_train,
        c_train, train_len, max_length, 
        clip_time, model_type = model_type)
    for score in SCORES:
        results['train_%s'%score] = s[score]
        for ii in range(k_clip):
            results['t_train_%s'%score][ii] = s_t[ii][score]
    print('train p = %0.3f' %s['p'])
    '''
    results on test data
    '''
    s, s_t, y, y_hat, c = dnn_score(
        model, X_test, y_test,
        c_test, test_len, max_length, 
        clip_time, model_type = model_type)
    for score in SCORES:
        results['test_%s'%score] = s[score]
        for ii in range(k_clip):
            results['t_test_%s'%score][ii] = s_t[ii][score]
    print('test p = %0.3f' %s['p'])

    results['y'] = y
    results['y_hat'] = y_hat
    results['c'] = c

    return results

def run(args):
    _info(args.bhv)
    _info(args.subnet)
    # set to regression mode
    args.mode = 'reg'
    # use data from all participants
    args.cutoff = 0.5
    '''
    get dataframe
    '''
    # all-but-subnetwork (invert_flag)
    if 'minus' in args.subnet:
        args.invert_flag = True

    res_path = (RES_DIR +
                '/roi_%d_net_%d' %(args.roi, args.net) +
                '_nw_%s' %(args.subnet) + 
                '_bhv_%s' %(args.bhv) +
                '_trainsize_%d' %(args.train_size) + 
                '_kfold_%d_k_hidden_%d' %(args.k_fold, args.k_hidden) +
                '_k_layers_%d_batch_size_%d' %(args.k_layers, args.batch_size) +
                '_num_epochs_%d_z_%d.pkl' %(args.num_epochs, args.zscore))

    if not os.path.isfile(res_path):
        df, bhv_df = _bhv_reg_df(args)
        results = {}
        results['train_mode'] = _train(df, bhv_df, args)
        results['test_mode'] = _test(df, bhv_df, args)
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
    
    # behavioral parameters
    parser.add_argument('-b', '--bhv', type=str,
        default='ListSort_Unadj',
        help='behavioral measure: PMAT24_A_CR, ListSort_Unadj (default), ... ')
    
    # preprocessing
    parser.add_argument('--zscore', type=int,
        default=1, help='zscore = 1 (default) or 0')
    
    # training parameters
    parser.add_argument('-k', '--k_fold', type=int,
        default=5, help='number of folds (default: 5) for cross validation')
    parser.add_argument('--k_hidden', type=int,
        default=32, help='size of hidden state (default: 32)')
    parser.add_argument('--k_layers', type=int,
        default=1, help='no. of gru layers (default: 1)')
    parser.add_argument('--batch_size', type=int,
        default=30, help='batch size for training (default: 30)')
    parser.add_argument('--num_epochs', type=int,
        default=20, help='no. of epochs for training (default: 20)')
    parser.add_argument('--train_size', type=int,
        default=100, help='number of participants in training data (default: 100)')
    
    # model hyperparams
    parser.add_argument('-l2','--l2', type=float,
        default=0.0, help='l2 penalty (default: 0.0)')
    parser.add_argument('--dropout', type=float,
        default=0.000001, help='dropout rate (default: 1e-6)')
    parser.add_argument('-lr','--lr', type=int,
        default=0.001, help='learning rate (default: 0.001)')
    
    args = parser.parse_args()
    
    run(args)
