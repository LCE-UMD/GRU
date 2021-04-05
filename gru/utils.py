import numpy as np
import tensorflow as tf
import pywt

def shuffle_ts(X, X_len):
    '''
    Shuffling clip time series
    
    Parameters:
    X: tensor (batch x time x feature)
    X_len: timeseries length tensor (batch x 1)
    
    Returns:
    X_copy: shuffled timeseries tensor (batch x time x feature)
    '''
    
    X_copy = tf.identity(X)
    X_copy = X_copy.numpy()
    # create mask to ignore padding
    mask = X_copy == 0.0
    # Go thru every example in the batch
    for ii in range(X_copy.shape[0]):
        unpadded_ts = X_copy[ii,:X_len[ii],:]
        # Take wavelet transform
        coeffs = pywt.wavedec(unpadded_ts,'db2',level=2,axis=0)
        # reconstruct the ts and assert if it is close to orig ts
        recon_ts = pywt.waverec(coeffs, 'db2', axis = 0)
        tf.debugging.assert_near(unpadded_ts,recon_ts[:X_len[ii]])
        # get perm coeffs
        perm_coeffs = permute(coeffs)
        # construct shuffled timeseries
        perm_ts = pywt.waverec(perm_coeffs, 'db2', axis = 0)
        X_copy[ii,:X_len[ii],:] = perm_ts[:X_len[ii],:]
        
    return tf.convert_to_tensor(X_copy,dtype='float32')

def permute(coeffs):
    '''
    Shuffles the wavelet transform coefficients of a timeseries
    
    Parameters:
    coeffs: list of the wavelet coefficients. Example (cA, cD1, cD2 ...cDn)
    
    Returns:
    perm_coeffs: list of the wavelet coefficiets in which only detail (cDn)
                    coefficients are permuted
    '''
    permuted_coeffs = []
    for coeff in coeffs:
        coeff_copy = coeff.copy()
        np.random.shuffle(coeff_copy)
        permuted_coeffs.append(coeff_copy)
    return permuted_coeffs


################# functions below can be deleted ################
def shuffle_ts_old(X,X_len):
    '''
    Suffling of the clip timeseries
    '''
    # make copy of X
    X_copy = tf.identity(X)
    X_copy = X_copy.numpy()
    # create mask to ignore padding
    mask = X_copy == 0.0
    # Go thru every example in the batch shuffle the timeseries excluding the padding
    for ii in range(X_copy.shape[0]):
        unpadded_example = X_copy[ii,:X_len[ii],:]
        X_copy[ii,:X_len[ii],:] = tf.random.shuffle(unpadded_example)
        
        # Assert the only temporal order is shuffled.
        # Raise error if the two timeseries have different set of values
        assert set(X_copy[ii,:,0]) == set(X[ii,:,0].numpy())
        # Raise error if the two timeseries are equal (not shuffled)
        np.testing.assert_raises(AssertionError, 
                                 np.testing.assert_array_equal, 
                                 X[ii,:,:].numpy(),
                                 X_copy[ii,:,:])
        
    return tf.convert_to_tensor(X_copy,dtype='float32')