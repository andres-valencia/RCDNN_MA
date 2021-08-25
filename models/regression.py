import tensorflow as tf
import numpy as np
from scipy.io import loadmat
from sklearn.preprocessing import StandardScaler
from scipy.io import savemat
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import OneHotEncoder
from scipy.stats import mode 
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA

from sklearn.base import  BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.model_selection import StratifiedShuffleSplit


def ook(t):
  lb = LabelBinarizer()
  y_ook = lb.fit_transform(t)  

  if len(np.unique(t))==2:
    y_ook = np.concatenate((1-y_ook.astype(bool), y_ook), axis = 1) 

  return y_ook

class Keras_MA_pi_kern(BaseEstimator, TransformerMixin):
  def __init__(self, K, R, P, epochs=30,batch_size=64,learning_rate=1e-3,optimizer='RMS',
               validation_split=0.3, verbose=1, ratio=1.2, scale=0.1, seed=None, ratio_lin=0.75,
               l1=0.01, l2=0.01, clfout=100, dropout=True, BN=True):
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate=learning_rate 
        self.validation_split = validation_split
        self.verbose = verbose
        self.optimizer = optimizer
        self.ratio = ratio
        self.K = 1
        self.R = R
        self.P = P
        self.scale = scale
        self.l1 = l1
        self.l2 = l2
        self.seed = seed
        self.clfout = clfout
        self.ratio_lin = ratio_lin
        self.dropout = dropout
        self.BN = BN

        
  def fit(self, X, t):
    #lb = LabelBinarizer()
    #lb.fit(X[:,-1])
    #N = X.shape[0]
    #y = np.zeros([N, self.K, self.R])
    #for i in range(N):
    #  y[i,:,:] = binarize(X[i,self.P:],lb).T
    y = t

    if self.optimizer == "Adam":
        opt = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
    elif self.optimizer == "SGD":
        opt = tf.keras.optimizers.SGD(learning_rate=self.learning_rate)
    elif self.optimizer == "RMS":
        opt = tf.keras.optimizers.RMSprop(learning_rate=self.learning_rate)
    else:
        opt=self.optimizer

    
    ###acomodar arquitectura de red###### 
    tf.keras.backend.clear_session()

    inputX1 = tf.keras.layers.Input(shape=(X.shape[1]), name='X1')
    if self.BN:
      inputX1b = tf.keras.layers.BatchNormalization()(inputX1)
    else:
      inputX1b = inputX1
    if self.dropout:
      inputX1b = tf.keras.layers.AlphaDropout(rate=0.2)(inputX1b)

    Q1 =  self.K ## num clases
    Q2 =  1 ## num anotadores
    Q31 = 8#self.ratio_lin
    Q32 = 50#self.ratio_lin
    Q33 = 20#self.ratio_lin

    initializer = tf.keras.initializers.GlorotNormal(seed=self.seed)

#    if self.w_init == 'PCA':
#      pca = PCA().fit(Xt).components_
#      initializer1 = tf.keras.initializers.Constant(value=pca[:Q3,:].T)
#    elif self.w_init == 'inv':
#      enc = OneHotEncoder()
#      ymv = mode(y2,1)[0]
#      yb = enc.fit_transform(ymv).toarray()
#      Q3 = self.K
#      w_init = np.linalg.pinv(Xt).dot(yb)
#      initializer1 = tf.keras.initializers.Constant(value=w_init)
#    else:
#      initializer1 = initializer


    X1_e1 = tf.keras.layers.Dense(Q31,activation='linear',name='Filt_X1',  bias_initializer='zeros', kernel_initializer=initializer,
                              kernel_regularizer=tf.keras.regularizers.l1_l2(l1=self.l1,l2=self.l2))(inputX1b)
    if self.BN:
      X1_e1 = tf.keras.layers.BatchNormalization()(X1_e1)
    if self.dropout:
      X1_e1 = tf.keras.layers.AlphaDropout(rate=0.2)(X1_e1)
    
    X1_e2 = tf.keras.layers.Dense(self.clfout,activation='relu',name='Clf_X1',  bias_initializer='zeros', kernel_initializer=initializer,
                              kernel_regularizer=tf.keras.regularizers.l1_l2(l1=self.l1,l2=self.l2))(X1_e1)                         
    if self.BN:
      X1_e2 = tf.keras.layers.BatchNormalization()(X1_e2)
    if self.dropout:
      X1_e2 = tf.keras.layers.AlphaDropout(rate=0.2)(X1_e2)

    #Clf = tf.keras.layers.Dense(self.clfout,activation='tanh',name='Clf',  bias_initializer='zeros', kernel_initializer=initializer,
    #                          kernel_regularizer=tf.keras.regularizers.l1_l2(l1=self.l1,l2=self.l2))(concX)                         
    #if self.BN:
    #  Clf = tf.keras.layers.BatchNormalization()(Clf)
    #if self.dropout:
    #  Clf = tf.keras.layers.AlphaDropout(rate=0.2)(Clf)

    pred = tf.keras.layers.Dense(Q1,activation='linear',name='Y_est',  bias_initializer='zeros', kernel_initializer=initializer,
                              kernel_regularizer=tf.keras.regularizers.l1_l2(l1=self.l1,l2=self.l2))(X1_e2)
    

    #out = tf.keras.layers.concatenate([pred, conf],name='output')

    self.model = tf.keras.Model(inputs=inputX1, outputs=pred)
    self.model.compile(loss="mean_squared_error",
              optimizer=opt) #f1, precision, recall, crossentropy
            

    self.history = self.model.fit(x = X, y = y,
                        epochs=self.epochs,batch_size=self.batch_size,
                        validation_split=self.validation_split,  
                        verbose = self.verbose)
    
  def plot_history_loss(self):
      plt.plot(self.history.history['loss'],label='loss')
      #plt.plot(self.history.history['val_loss'],label='val_loss')
      plt.legend()
      return
