from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from tensorflow import keras

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

def _get_true_class_prob(y,y_probs,seq_len):
    
    y_prob_true = defaultdict(list)
    
    for i in range(y.shape[0]):
        if int(y[i,0]) not in y_prob_true:
            y_prob_true[int(y[i,0])] = []
            
        y_prob_true[int(y[i,0])].append(y_probs[i,:seq_len[i],int(y[i,0])])
    return y_prob_true


def _get_t_acc(y_hat, y, k_time):
    '''
    accuracy as f(time)
    '''
    #print('y_hat:', len(y_hat))
    #print('y:', len(y))
    a = np.zeros(k_time)
    for ii in range(k_time):
        y_i = y[ii::k_time]
        #print('y_i:', len(y_i))
        y_hat_i = y_hat[ii::k_time]
        correct = [1 for p, q in zip(y_i, y_hat_i) if p==q]
        a[ii] = sum(correct)/len(y_i)
        
    return a

def _get_confusion_matrix(y, predicted):
    '''
    confusion matrix per class
    '''
    y, p = y, predicted

    return confusion_matrix(y, p)


def _gru_acc(model, X, y, clip_time):
    '''
    masked accuracy for gru
    '''
    # mask to ignore padding
    mask = model.layers[0].compute_mask(X)

    # predicted labels
    '''
    WARNING:tensorflow:From /home/climbach/06-lstm_shared-master/gru/cc_utils.py:81: 
    Sequential.predict_classes (from tensorflow.python.keras.engine.sequential) 
    is deprecated and will be removed after 2021-01-01.
    
    Instructions for updating:
    Please use instead:* `np.argmax(model.predict(x), axis=-1)`,
    if your model does multi-class classification  (e.g. if it uses a `softmax` last-layer activation).
    * `(model.predict(x) > 0.5).astype("int32")`, 
    if your model does binary classification   (e.g. if it uses a `sigmoid` last-layer activation)
    '''
    y_hat = model.predict_classes(X)

    # remove padded values
    # converts matrix to vec
    y_hat = y_hat[mask==True]
    y = y[mask==True]
    y = y.numpy()

    # accuracy for all t
    correct = (y_hat == y).sum().item()
    a = correct/(mask.numpy()==True).sum()

    # accuracy as a function of t
    k_class = len(clip_time)
    a_t = {}
    for ii in range(k_class):
        y_i = y[y==ii]
        y_hat_i = y_hat[y==ii]
        k_time = clip_time[ii]
        a_t[ii] = _get_t_acc(y_hat_i, y_i, k_time)

    c_mtx = _get_confusion_matrix(y,y_hat)
    
    return a, a_t, c_mtx

def _gru_test_acc(model, X, y, clip_time, k_sub):
    '''
    masked accuracy for gru
    '''
    # mask to ignore padding
    mask = model.layers[0].compute_mask(X)

    # predicted labels
    y_hat = model.predict_classes(X)

    # remove padded values
    # converts matrix to vec
    y_hat = y_hat[mask==True]
    y = y[mask==True]
    y = y.numpy()

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

def _gruenc_test_traj(model, X):
    '''
    mask trajectories for GRUEncoder
    X is of shape num_samples x time x ROIs
    '''
    # mask to ignore padding
    mask = model.layers[0].compute_mask(X)
    
    # generate trajectories
    model.trainable = False
    new_model = keras.models.Sequential(model.layers[:-1])
    traj = new_model.predict(X)
    
    # squeeze traj if traj is of shape 1 x time x ROIs
    # i.e. num_samples = 1
    traj = np.squeeze(traj) # time x ROIS
    
    return traj

def _compute_saliency_maps(model, fMRI_sequence, target_class_idx):
    '''
    compute saliency maps:
    1> pass fMRI_sequence to get class scores
    2> backpropagate class score of target class
    3> return gradients w.r.t. inputs
    '''
    fMRI_sequence = tf.convert_to_tensor(fMRI_sequence)
    with tf.GradientTape() as tape:
        tape.watch(fMRI_sequence)
        probs = model(fMRI_sequence)[:, :, target_class_idx] # class probability score of the target class
    return tape.gradient(probs, fMRI_sequence).numpy()