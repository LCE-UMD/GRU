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

from dataloader import _clip_class_df, _clip_class_rest_df, K_RUNS
from gru.dataloader import _get_clip_seq as _get_seq
from gru.models import GRUEncoder
from gru.cc_utils import _get_true_class_prob, _gru_acc, _gru_test_acc, _gruenc_test_traj
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
    tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
except:
    # Invalid device or cannot modify virtual devices once initialized.
    pass

# results directory
RES_DIR = 'results/clip_gruencoder'
if not os.path.exists(RES_DIR):
    os.makedirs(RES_DIR)

K_SEED = 330

def _train(df,args,params):
    '''
    cross-validation results
    '''
    _info('Running grid search')
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

    # results dict init
    results = {}
    
    # mean accuracy across time
    results['train'] = np.zeros(args.k_fold)
    results['val'] = np.zeros(args.k_fold)
    
    # confusion matrices
    results['train_conf_mtx'] = np.zeros((args.k_class, args.k_class))
    results['val_conf_mtx'] = np.zeros((args.k_class, args.k_class))
    
    # per class temporal accuracy
    results['t_train'] = {}
    results['t_val'] = {}
    for ii in range(args.k_class):
        results['t_train'][ii] = np.zeros((args.k_fold, clip_time[ii]))
        results['t_val'][ii] = np.zeros((args.k_fold, clip_time[ii]))

    i_fold = 0
    kf = KFold(n_splits=args.k_fold, random_state=K_SEED)

    for train, val in kf.split(train_list):
        _info('fold: %d/%d' %(i_fold+1, args.k_fold))

        # ***between-subject train-val split
        train_subs = [train_list[ii] for ii in train]
        val_subs = [train_list[ii] for ii in val]

        # get train, val sequences
        X_train, train_len, y_train = _get_seq(df, 
            train_subs, args)
        X_val, val_len, y_val = _get_seq(df, 
            val_subs, args)

        '''
        train classifier
        '''
        then = time.time()
        model = GRUEncoder(X_train, args.gru_model_path,
                              k_layers=params['k_layers'], 
                              k_hidden=params['k_hidden'],
                              k_class = args.k_class)
        
        model.fit(X_train,y_train,epochs=args.num_epochs,
                  validation_split=0.2,
                  batch_size=args.batch_size,
                  verbose=1)
        print('--- train time =  %0.4f seconds ---' %(time.time() - then))
        
        trainable = np.sum([tf.reshape(params,-1).shape[0] for params in model.trainable_variables])
        print('Total trainable parameters: %i'%trainable)
        '''
        results on train data
        '''
        a, a_t, c_mtx = _gru_acc(model, X_train, y_train, clip_time)
        results['train'][i_fold] = a
        print('tacc = %0.3f' %a)
        for ii in range(args.k_class):
            results['t_train'][ii][i_fold] = a_t[ii]
        results['train_conf_mtx'] += c_mtx
        
        '''
        results on val data
        '''
        a, a_t, c_mtx = _gru_acc(model, X_val, y_val, clip_time)
        results['val'][i_fold] = a
        print('vacc = %0.3f' %a)
        for ii in range(args.k_class):
            results['t_val'][ii][i_fold] = a_t[ii]
        results['val_conf_mtx'] += c_mtx
        
        i_fold += 1
        
    return results

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

    # results dict init
    results = {}
    
    # mean accuracy across time
    results['train'] = np.zeros(len(test_list))
    results['val'] = np.zeros(len(test_list))

    # per class temporal accuracy
    results['t_train'] = {}
    results['t_test'] = {}
    for ii in range(args.k_class):
        results['t_train'][ii] = np.zeros(
            (len(test_list), clip_time[ii]))
        results['t_test'][ii] = np.zeros(
            (len(test_list), clip_time[ii]))
        
        
    results_prob = {}    
    for method in 'train test'.split():
        results_prob[method] = {}
        for measure in 'acc t_prob'.split():
            results_prob[method][measure] = {}
                
    '''
    init model
    '''
    # get train, test sequences
    X_train, train_len, y_train = _get_seq(df, 
        train_list, args)
    X_test, test_len, y_test = _get_seq(df, 
        test_list, args)

    '''
    train classifier
    '''
    then = time.time()
    model = GRUEncoder(X_train, args.gru_model_path,
                              k_layers=params['k_layers'], 
                              k_hidden=params['k_hidden'],
                              k_dim=args.k_dim,
                              k_class = args.k_class)
        
    model.fit(X_train,y_train,epochs=args.num_epochs,
              validation_split=0.2,
              batch_size=args.batch_size,
              verbose=1)
        
    print('--- train time =  %0.4f seconds ---' %(time.time() - then))
    
    '''
    results on train data
    '''
    a, a_t, c_mtx = _gru_test_acc(model, X_train, y_train,
                                  clip_time, len(train_list))
    results['train'] = a
    print('tacc = %0.3f' %np.mean(a))
    for ii in range(args.k_class):
        results['t_train'][ii] = a_t[ii]
    results['train_conf_mtx'] = c_mtx
    
    # train temporal probs
    results_prob['train']['acc'] = model.evaluate(X_train,y_train)[1]
    X_train_probs= model.predict(X_train)
    results_prob['train']['t_prob'] = _get_true_class_prob(y_train, X_train_probs, train_len)
    
    '''
    results on test data
    '''
    a, a_t, c_mtx = _gru_test_acc(model, X_test, y_test,
                                  clip_time, len(test_list))
    results['test'] = a
    print('sacc = %0.3f' %np.mean(a))
    for ii in range(args.k_class):
        results['t_test'][ii] = a_t[ii]
    results['test_conf_mtx'] = c_mtx
    
    # test temporal probs
    results_prob['test']['acc'] = model.evaluate(X_test,y_test)[1]
    X_test_probs= model.predict(X_test)
    results_prob['test']['t_prob'] = _get_true_class_prob(y_test, X_test_probs, test_len)
    
    results['train_subs'] = train_list
    results['test_subs'] = test_list
            
    return results, results_prob, model



def _get_trajectories(df, model, args):
    '''
    forward pass data
    save trajectories
    '''
    for kk in range(args.k_dim):
        df.loc[:, 'dim_%d'%(kk)] = np.nan
    dims = [ii for ii in df.columns if 'dim' in ii]
    
    k_class = len(np.unique(df['y']))
    print('number of unique sequences = %d' %k_class)
    subject_list = np.unique(df['Subject'])
    
    # forward pass
    features = [ii for ii in df.columns if 'feat' in ii]
    
    for subject in subject_list:
        for i_class in range(k_class):
    
            # don't handle test retest differently
            # labels based on full_run_to_df are different
            
            seq = df[(df['Subject']==subject) & 
                (df['y']==i_class)][features].values
            if args.zscore:
                # zscore each seq that goes into model
                seq = (1/np.std(seq))*(seq - np.mean(seq))

            # pad sequences    
            X = [seq]
            X_padded = tf.keras.preprocessing.sequence.pad_sequences(
                X, padding="post", dtype='float')
            
            # forward pass
            traj = _gruenc_test_traj(model, X_padded)

            df.loc[(df['Subject']==subject) & 
                        (df['y']==i_class), dims] = traj

    traj_df = df[['Subject', 'timepoint', 'y'] + dims]

    return traj_df



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
        
        gru_model_path = mod_path.replace('gruencoder', 'gru')
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
        df = _clip_class_df(args)
        print('data loading time: %.2f seconds' %(time.clock()-start))
        
        if len(param_grid) == 1:
            results, results_prob, model = _test(df,args,param_grid[0])
            print('TESTING DONE :)')
            # forward pass data and save encodings
            #traj_df = {}
            #for run in range(K_RUNS):
            #    df = _clip_class_rest_df(args, run)
            #    traj_df[run] = _get_trajectories(df, model, args)
            
            # save results
            with open(res_path, 'wb') as f:
                pickle.dump([results, results_prob], f)
            # save model
            #model.save(mod_path)
                
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
