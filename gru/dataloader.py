import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from gru.utils import shuffle_ts
import random

K_RUNS = 4

def _get_clip_seq(df, subject_list,args, shuffle = False, label_shuffle=False):
    '''
    return:
    X: input seq (batch_size x time x feat_size)
    y: label seq (batch_size x time)
    X_len: len of each seq (batch_size x 1)
    batch_size <-> number of sequences
    time <-> max length after padding
    '''
    features = [ii for ii in df.columns if 'feat' in ii]

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
#                     if shuffle == True:
#                         np.random.shuffle(seq)
                    X.append(seq)
                    y.append(label_seq)
            else:
                seq = df[(df['Subject']==subject) & 
                    (df['y'] == i_class)][features].values
                label_seq = df[(df['Subject']==subject) & 
                    (df['y'] == i_class)]['y'].values
                if args.zscore:
                    # zscore each seq that goes into model
                    seq = (1/np.std(seq))*(seq - np.mean(seq))
#                 if shuffle == True:
#                     np.random.shuffle(seq)
                X.append(seq)
                y.append(label_seq)
                
    X_len = tf.convert_to_tensor([len(seq) for seq in X])

    X_padded = tf.keras.preprocessing.sequence.pad_sequences(
        X, padding="post",
        dtype='float'
    )
    if label_shuffle == True:
        random.shuffle(y)
#     if shuffle == True:
#         X_padded = shuffle_ts(X_padded, X_len)

    y_padded = tf.keras.preprocessing.sequence.pad_sequences(
        y, padding="post",
        dtype='float'
    )
    y = np.array([array[0] for array in y])
    
    return tf.convert_to_tensor(X_padded,dtype='float32'), X_len ,tf.convert_to_tensor(y_padded,dtype='float32')


def _get_bhv_seq(df, subject_list, args, label_shuffle=False):
    '''
    return:
    X: input seq (batch_size x time x feat_size)
    y: label seq (batch_size x time)
        in {0, 1, ..} if args.mode=='class'
        in R if args.mode=='reg'
    c: clip seq (batch_size x time)
    X_len: len of each seq (batch_size x 1)
    batch_size <-> number of sequences
    time <-> max length after padding
    '''
    # optional arguments
    d = vars(args)

    # regression or classification
    if 'mode' not in d:
        args.mode = 'class'
    if args.mode=='class':
        label = 'y'
    elif args.mode=='reg':
        label = args.bhv

    # permutation test
    if 'shuffle' not in d:
        args.shuffle = False
    if args.shuffle:
        # different shuffle for each iteration
        np.random.seed(args.i_seed)
        # get scores for all participants without bhv_df
        train_label = df[(df['Subject'].isin(subject_list)) &
            (df['c']==1) & (df['timepoint']==0)][label].values
        np.random.shuffle(train_label) # inplace

    k_clip = len(np.unique(df['c']))
    features = [ii for ii in df.columns if 'feat' in ii]

    X = []
    y = []
    c = []

    for ii, subject in enumerate(subject_list):
        for i_clip in range(k_clip):

            if i_clip==0: #handle test retest differently
                seqs = df[(df['Subject']==subject) & 
                    (df['c'] == 0)][features].values
                if args.shuffle:
                    label_seqs = np.ones(seqs.shape[0])*train_label[ii]
                else:
                    label_seqs = df[(df['Subject']==subject) & 
                        (df['c'] == 0)][label].values
                clip_seqs = df[(df['Subject']==subject) & 
                    (df['c'] == 0)]['c'].values

                k_time = int(seqs.shape[0]/K_RUNS)
                for i_run in range(K_RUNS):
                    seq = seqs[i_run*k_time:(i_run+1)*k_time, :]
                    label_seq = label_seqs[i_run*k_time:(i_run+1)*k_time]
                    clip_seq = clip_seqs[i_run*k_time:(i_run+1)*k_time]
                    if args.zscore:
                        # zscore each seq that goes into model
                        seq = (1/np.std(seq))*(seq - np.mean(seq))

                    X.append(tf.convert_to_tensor(seq,dtype='float32'))
                    if args.mode=='class':
                        y.append(tf.convert_to_tensor(label_seq,dtype=np.int32))
                    elif args.mode=='reg':
                        y.append(tf.convert_to_tensor(label_seq,dtype='float32'))
                    c.append(tf.convert_to_tensor(clip_seq,dtype=np.int32))
            else:
                seq = df[(df['Subject']==subject) & 
                    (df['c'] == i_clip)][features].values
                if args.shuffle:
                    label_seq = np.ones(seq.shape[0])*train_label[ii]
                else:
                    label_seq = df[(df['Subject']==subject) & 
                        (df['c'] == i_clip)][label].values
                clip_seq = df[(df['Subject']==subject) & 
                    (df['c'] == i_clip)]['c'].values
                if args.zscore:
                    # zscore each seq that goes into model
                    seq = (1/np.std(seq))*(seq - np.mean(seq))

                X.append(tf.convert_to_tensor(seq,dtype='float32'))
                if args.mode=='class':
                    y.append(tf.convert_to_tensor(label_seq,dtype=np.int32))
                elif args.mode=='reg':
                    y.append(tf.convert_to_tensor(label_seq,dtype='float32'))
                c.append(tf.convert_to_tensor(clip_seq,dtype=np.int32))

    X_len = tf.convert_to_tensor([len(seq) for seq in X],dtype=np.int32)
    if label_shuffle == True:
        random.shuffle(y)

    # pad sequences
    X = tf.keras.preprocessing.sequence.pad_sequences(X, padding="post",value=0.,dtype='float32')
    if args.mode == 'class':
        y = tf.keras.preprocessing.sequence.pad_sequences(y, padding="post",value=0.,dtype='int32')
    elif args.mode == 'reg':
        y = tf.keras.preprocessing.sequence.pad_sequences(y, padding="post",value=0.,dtype='float32')
    c = tf.keras.preprocessing.sequence.pad_sequences(c, padding="post",value=0)

    return X, X_len, y, c

## Functions to load emoprox data from here on ##

def _emo_class_df(args):
    '''
    data for approach-retreat classification

    args.input_data: path/to/seg/dataset

    save each timepoint as feature vector
    append class label based on clip

    return:
    pandas df
    '''
    if args.near_miss:
        with open(args.input_data,"rb") as f:
            dataset = pickle.load(f)

        n_vox = dataset['CON031']['data'].shape[1]
        features = ['feat_%i'%ii for ii in range(n_vox)]
        df = pd.DataFrame(columns=['Subject','timepoint']+features+['y'])
        for subj in dataset:
            for trial in range(dataset[subj]['data'].shape[-1]):
                features = ['feat_%i'%ii for ii in range(n_vox)]
                tmp_df = pd.DataFrame(dataset[subj]['data'][:,:,trial],
                                      columns=features,
                                      index=np.arange(dataset[subj]['data'].shape[0]))
                tmp_df['Subject'] = int(subj.replace('CON',''))
                tmp_df['TimePoint'] = np.arange(14)
                tmp_df['timepoint'] = list(range(7)) + list(range(7))
                tmp_df['y'] = tmp_df.TimePoint.apply(lambda tp: 1 if tp < 7 else 0)
                df = pd.concat([df,tmp_df],axis=0,ignore_index=True)
        df.timepoint = df.timepoint.astype(int)
        df.TimePoint = df.TimePoint.astype(int)
        df.y = df.y.astype(int)
    else:
        with open(args.input_data,"rb") as f:
            df = pickle.load(f)
    return df


def _get_emo_seq(df, subject_list, args, label_shuffle=False):
    '''
    return:
    X: input seq (batch_size x time x feat_size)
    y: label seq (batch_size x time)
    X_len: len of each seq (batch_size x 1)
    batch_size <-> number of sequences
    time <-> max length after padding
    '''
    features = [ii for ii in df.columns if 'feat' in ii]

    X = []
    y = []
    for subject in subject_list:
        subj_df = df[df['Subject']==subject]
        subj_df.reset_index(inplace=True)
        trials = np.split(subj_df,subj_df[subj_df['timepoint']==0].index)[1:]
        for ii,trial in enumerate(trials):
            seq = trial[trial['Subject'] == subject][features].values
            label_seq = trial[trial['Subject']==subject]['y'].values
            if args.zscore:
                # zscore each seq that goes into model
                seq = (1/np.std(seq))*(seq - np.mean(seq))

            X.append(seq)
            y.append(label_seq)
    if label_shuffle == True:
        random.shuffle(y)
    X_len = tf.convert_to_tensor([len(seq) for seq in X])

    # pad sequences
    X_padded = tf.keras.preprocessing.sequence.pad_sequences(
        X, padding="post",
        dtype='float'
    )

    y_padded = tf.keras.preprocessing.sequence.pad_sequences(
        y, padding="post",
        dtype='float'
    )
    y = np.array([array[0] for array in y])
    
    return tf.convert_to_tensor(X_padded,dtype='float32'), X_len ,tf.convert_to_tensor(y_padded,dtype='float32')
