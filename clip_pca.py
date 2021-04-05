'''
model: logistic regression on PC-transformed inputs
task: predict clip (15 way classifier)
data: all runs used together
input to model: clip time series/seq
output: label time series
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
from gru.models import LogReg
'''
Helpers
'''
from dataloader import _clip_class_df
from gru.dataloader import _get_clip_seq as _get_seq
from utils import _info
from gru.cc_utils import _gru_acc as _ff_acc
from gru.cc_utils import _gru_test_acc as _ff_test_acc

# results directory
RES_DIR = 'results/clip_pca'
if not os.path.exists(RES_DIR):
    os.makedirs(RES_DIR)

K_SEED = 330
    
def _get_pc(df, train_list, test_list, args):
    
    pc_df = df[['Subject', 'timepoint', 'y']].copy()

    features = [ii for ii in df.columns if 'feat' in ii]
    train_df = df[df['Subject'].isin(train_list)]
    test_df = df[df['Subject'].isin(test_list)]

    X_train = train_df[features].values 
    X_test = test_df[features].values
    
    then = time.time()
    #
    pca = PCA(n_components=args.k_dim)
    X_pc_train = pca.fit_transform(X_train)
    X_pc_test = pca.transform(X_test)
    print('--- pca transform time =  {:0.4f} seconds ---'.format(
        time.time() - then))

    for kk in range(args.k_dim):
        pc_df.loc[:, 'feat_%d'%(kk)] = np.nan
    # overwrite features for get_clip_seq
    features = [ii for ii in pc_df.columns if 'feat' in ii]
    
    pc_df.loc[pc_df['Subject'].isin(train_list), features] = X_pc_train
    pc_df.loc[pc_df['Subject'].isin(test_list), features] = X_pc_test
              
    return pc_df
    
def _test(df,args):
    '''
    test subject results
    view only for best cross-val parameters
    '''
    _info('test mode')

    # get X-y from df
    subject_list = df['Subject'].unique()
    train_list = subject_list[:args.train_size]
    test_list = subject_list[args.train_size:]

    pc_df = _get_pc(df, train_list, test_list, args)

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
    '''
    init model
    '''

    # get train, test sequences
    X_train, train_len, y_train = _get_seq(pc_df, 
        train_list, args)
    X_test, test_len, y_test = _get_seq(pc_df, 
        test_list, args)

    '''
    train classifier
    '''
    then = time.time()
    model = LogReg(k_dim=args.k_dim, 
                   k_class=args.k_class)

    model.fit(X_train,y_train,epochs=args.num_epochs,
              validation_split=0.2,
              batch_size=args.batch_size,
              verbose=1)

    print('--- train time =  %0.4f seconds ---' %(time.time() - then))

    '''
    results on train data
    ff_test_acc works for logreg
    '''
    a, a_t, c_mtx = _ff_test_acc(model, X_train, y_train,
                                 clip_time, len(train_list))
    results['train'] = a
    print('tacc = %0.3f' %np.mean(a))
    for ii in range(args.k_class):
        results['t_train'][ii] = a_t[ii]
    '''
    results on test data
    '''
    a, a_t, c_mtx = _ff_test_acc(model, X_test, y_test, 
                                 clip_time, len(test_list))
    results['test'] = a
    print('sacc = %0.3f' %np.mean(a))
    for ii in range(args.k_class):
        results['t_test'][ii] = a_t[ii]

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
        '/%s_%d_net_%d' %(args.roi_name, args.roi, args.net) + 
        '_nw_%s' %(args.subnet) +
        '_trainsize_%d' %(args.train_size) +
        '_kdim_%d_batch_size_%d' %(args.k_dim, args.batch_size) +
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
    parser.add_argument('-rn','--roi_name', type=str,
                       default='roi', help='defualt: "roi"')
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
        default=5, help='number of folds for cross validation')
    parser.add_argument('--k_dim', type=int,
        default=3, help='number of dimensions in low-dim projection')
    parser.add_argument('--batch_size', type=int,
        default=32, help='batch size for training')
    parser.add_argument('--num_epochs', type=int,
        default=50, help='no. of epochs for training')
    parser.add_argument('--train_size', type=int,
        default=100, help='number of participants in training data')
    
    args = parser.parse_args()
    
    run(args)

    print('finished!')