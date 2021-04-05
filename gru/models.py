import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa
from sklearn.metrics import r2_score

from functools import partial

'''
classification
'''
# GRU clip classifier
def GRUClassifier(X, k_layers=1, k_hidden=32, k_class=15,
                  l2=0.001, dropout=1e-6, lr=0.006, seed=42):
    
    """
    Parameters
    ---------
    X: tensor (batch x time x feat)
    k_layers: int, number of hidden layers
    k_hidden: int, number of units
    k_class: int, number of classes
    
    Returns
    -------
    model: complied model
    """
    
    tf.random.set_seed(seed)
    regularizer = keras.regularizers.l2(l2)
    CustomGRU = partial(keras.layers.GRU,
                            kernel_regularizer=regularizer,
                            dropout=dropout,
                            recurrent_dropout=dropout
                           )
    '''
    For masking, refer: 
        https://www.tensorflow.org/guide/keras/masking_and_padding
        https://gist.github.com/ragulpr/601486471549cfa26fe4af36a1fade21
    '''
    input_layers = [layers.Masking(mask_value=0.0, 
                                   input_shape = [None, X.shape[-1]])]
    
    hidden_layers = []
    for ii in range(k_layers):
        hidden_layers.append(CustomGRU(k_hidden,return_sequences=True))
        
    output_layer = [layers.TimeDistributed(layers.Dense(k_class,activation='softmax'))]
    
    optimizer = keras.optimizers.Adam(lr=lr)
    
    model = keras.models.Sequential(input_layers+hidden_layers+output_layer)
    
    model.compile(loss='sparse_categorical_crossentropy',
                      optimizer=optimizer,metrics=['sparse_categorical_accuracy'])
    
    return model

# GRU encoder

def GRUEncoder(X, gru_model_path, k_layers=1, k_hidden=32, k_dim = 3,
               k_class = 15,
               l2=0.001, dropout=1e-6, lr=0.006, seed=42):
    
    '''
    GRU Encoder: classification after supervised dim reduction
    
    Parameters
    ----------
    X: tensor (batch x time x feat)
    k_layers: int, number of hidden layers
    k_hidden: int, number of units
    k_dim: int, reduce to k_dim
    k_class: int, number of classes
    
    Returns
    -------
    model: complied model
    '''
    
    tf.random.set_seed(seed)
    regularizer = keras.regularizers.l2(l2)
    
    ''' 
    Transfer Learning
    -----------------
    Using pretrained gru model for finetuning DR_layer 
    '''
    gru_model = keras.models.load_model(gru_model_path)
    gru_model.trainable = False
    
    '''
    For masking, refer: 
        https://www.tensorflow.org/guide/keras/masking_and_padding
        https://gist.github.com/ragulpr/601486471549cfa26fe4af36a1fade21
    '''
    input_layers = [layers.Masking(mask_value=0.0, 
                                   input_shape = [None, X.shape[-1]])]
    
    hidden_layers = [gru_model.layers[1]]
        
    DR_layer = [layers.TimeDistributed(layers.Dense(k_dim,activation='linear'))]
    output_layer = [layers.TimeDistributed(layers.Dense(k_class,activation='softmax'))]
    
    optimizer = keras.optimizers.Adam(lr=lr)
    
    model = keras.models.Sequential(input_layers +
                                    hidden_layers +
                                    DR_layer +
                                    output_layer)
    
    model.compile(loss='sparse_categorical_crossentropy',
                      optimizer=optimizer,metrics=['sparse_categorical_accuracy'])
    
    return model

# GRU Decoder

def GRUDecoder(X, Y, k_layers=1,
                  l2=0, dropout=0, lr=0.001,seed=42):
    
    """
    Parameters
    ---------
    X: tensor (batch x time x feat)
    k_layers: int, number of hidden layers
    k_hidden: int, number of units
    k_class: int, number of classes
    
    Returns
    -------
    model: complied model
    """
    
    tf.random.set_seed(seed)
    regularizer = keras.regularizers.l2(l2)
    CustomGRU = partial(keras.layers.GRU,
                            kernel_regularizer=regularizer,
                            dropout=dropout,
                            recurrent_dropout=dropout
                           )
    input_layers = [layers.Masking(mask_value=0.0, 
                                   input_shape = [None, X.shape[-1]])]
    
    hidden_layers = []
    
    for ii in range(k_layers):
        hidden_layers.append(CustomGRU(Y.shape[-1],return_sequences=True))
    
    optimizer = keras.optimizers.Adam(lr=lr)
    
    model = keras.models.Sequential(input_layers+hidden_layers)
    
    model.compile(loss='mse', optimizer=optimizer)
    return model


'''
classifier (FeedForward)
k_feat: (k_hidden:)*k_layers: k_class
'''
def FFClassifier(X,k_hidden,k_layers,k_class,seed=42):
    '''
    Feed-forward network classifier
    
    Parameters
    ----------
    X: tensor (batch x time x feat)
    k_layers: int, number of hidden layers
    k_hidden: int, number of units
    k_class: int, number of classes
    
    Returns
    -------
    model: complied model
    '''
        
    tf.random.set_seed(seed)
    input_layers = [layers.Masking(mask_value=0.0, input_shape = [X.shape[-2], X.shape[-1]])]
    
    hidden_layers = []
    for ii in range(k_layers):
        hidden_layers.append(layers.Dense(k_hidden,activation='relu'))
    
    output_layer = [layers.Dense(k_class,activation='softmax')]

    model = keras.models.Sequential(input_layers+hidden_layers+output_layer)
    
    optimizer = keras.optimizers.Adam()
    model.compile(loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                      optimizer=optimizer,metrics=['sparse_categorical_accuracy'])
    return model

'''
classifier (TCN)
k_hidden: filters, k_wind: kernel_size
'''
def TCNClassifier (X, k_hidden, k_wind, k_class,seed=42):
    '''
    TCN classifier
    
    Parameters
    ----------
    X: tensor (batch x time x feat)
    k_hidden: int, number of filters
    k_wind: int, kernel size
    k_class: int, number of classes
    
    Returns
    -------
    model: complied model
    '''
    
    tf.random.set_seed(seed)
    input_layers = [layers.Masking(mask_value=0.0, 
                                   input_shape = [None, X.shape[-1]])]
    hidden_layers = [layers.Conv1D(filters=k_hidden,kernel_size=k_wind,
                                   strides=1,padding='same',activation="relu")]
    
    output_layer = [layers.TimeDistributed(layers.Dense(k_class,activation='softmax'))]

    model = keras.models.Sequential(input_layers+hidden_layers+output_layer)
    
    optimizer = keras.optimizers.Adam()
    model.compile(loss='sparse_categorical_crossentropy',
                      optimizer=optimizer,metrics=['sparse_categorical_accuracy'])
    return model

'''
classifier (LogReg)
k_feat: k_class
'''
def LogReg(k_dim=3,k_class=15,seed=42):
    '''
    Logistic regression classifier 
    
    Parameters
    ----------
    k_dim: int, number of input features
    k_class: int, number of classes
    
    Returns
    -------
    model: complied model
    '''
        
    tf.random.set_seed(seed)
    masking_layer = [
        layers.Masking(mask_value=0.0, input_shape=[None,k_dim])
    ]
    output_layer = [
        layers.Dense(k_class,activation='softmax')
    ]
    model = keras.models.Sequential(
        masking_layer + output_layer
    )
    
    optimizer = keras.optimizers.Adam()
    model.compile(loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                      optimizer=optimizer,metrics=['sparse_categorical_accuracy'])
    return model


'''
regression: GRU
'''
def GRURegressor(X,k_layers=1, k_hidden=32, 
                 l2=0, dropout=0, lr=0.001,seed=42):
    
    """
    GRU regressor for individual difference
    
    Parameters
    ---------
    X: tensor (batch x time x feat)
    k_layers: int, number of hidden layers
    k_hidden: int, number of units
    
    Returns
    -------
    model: complied model
    """
    tf.random.set_seed(seed)
    regularizer = keras.regularizers.l2(l2)
    CustomGRU = partial(keras.layers.GRU,
                            kernel_regularizer=regularizer,
                            dropout=dropout,
                            recurrent_dropout=dropout
                           )
    '''
    For masking, refer: 
        https://www.tensorflow.org/guide/keras/masking_and_padding
        https://gist.github.com/ragulpr/601486471549cfa26fe4af36a1fade21
    '''
    input_layers = [layers.Masking(mask_value=0.0, 
                                   input_shape = [None, X.shape[-1]])]
    
    hidden_layers = []
    for ii in range(k_layers):
        hidden_layers.append(CustomGRU(k_hidden,return_sequences=True))
        
    output_layer = [layers.TimeDistributed(layers.Dense(1,activation='linear'))]
    
    optimizer = keras.optimizers.Adam(lr=lr)
    
    model = keras.models.Sequential(input_layers+hidden_layers+output_layer)
    
    model.compile(loss='mse',
                      optimizer=optimizer)
    
    return model

'''
regression: FF
'''
def FFRegressor (X,k_hidden,k_layers,seed=42):
    
    """
    FF regressor for individual difference
    
    Parameters
    ---------
    X: tensor (batch x time x feat)
    k_layers: int, number of hidden layers
    k_hidden: int, number of units
    
    Returns
    -------
    model: complied model
    """
    
    tf.random.set_seed(seed)
    input_layers = [layers.Masking(mask_value=0.0, input_shape = [X.shape[-2], X.shape[-1]])]
    
    hidden_layers = []
    for ii in range(k_layers):
        hidden_layers.append(layers.Dense(k_hidden,activation='relu'))
    
    output_layer = [layers.Dense(1,activation='linear')]

    model = keras.models.Sequential(input_layers+hidden_layers+output_layer)
    
    optimizer = keras.optimizers.Adam()
    model.compile(loss='mse', optimizer=optimizer)
    return model


def TCNRegressor (X, k_hidden, k_wind, seed=42):
    '''
    TCN classifier
    
    Parameters
    ----------
    X: tensor (batch x time x feat)
    k_hidden: int, number of filters
    k_wind: int, kernel size
    
    Returns
    -------
    model: complied model
    '''
    
    tf.random.set_seed(seed)
    input_layers = [layers.Masking(mask_value=0.0, 
                                   input_shape = [None, X.shape[-1]])]
    hidden_layers = [layers.Conv1D(filters=k_hidden,kernel_size=k_wind,
                                   strides=1,padding='same',activation='relu')]
    
    output_layer = [layers.TimeDistributed(layers.Dense(1,activation='linear'))]

    model = keras.models.Sequential(input_layers+hidden_layers+output_layer)
    
    optimizer = keras.optimizers.Adam()
    
    model.compile(loss='mse',optimizer=optimizer)
    return model