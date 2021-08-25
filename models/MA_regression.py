
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


def data_partitionMA(X,Y,t,ind,frac,seed):
  import math
  """ Function to partition data into train, test
  ind  --> matrix with shufled index (w,N) where w are the different possible random states
  frac --> fraction of data for training
  seed --> fixed random state for reproducibility
  """
  N = X.shape[0]
  Xtrain = X[ind[seed][:math.ceil(N*frac)]]
  Ytrain = Y[ind[seed][:math.ceil(N*frac)]]
  ttrain = t[ind[seed][:math.ceil(N*frac)]]

  Xtest = X[ind[seed][math.ceil(N*frac):]]
  Ytest = Y[ind[seed][math.ceil(N*frac):]]
  ttest = t[ind[seed][math.ceil(N*frac):]]

  return Xtrain, Ytrain, ttrain, Xtest, Ytest, ttest

def ook(t):
  lb = LabelBinarizer()
  y_ook = lb.fit_transform(t)  

  if len(np.unique(t))==2:
    y_ook = np.concatenate((1-y_ook.astype(bool), y_ook), axis = 1) 

  return y_ook


def bin_Y(Y):
  aux = np.unique(Y)
  aux = aux[aux>-1]
  N = Y.shape[0] 
  K = len(aux)
  R = Y.shape[1]
  Ynew = np.zeros([N,K,R])

  for r in range(R):
    aux = Y[:,r]
    for k in range(K):
      if K==2:
        Ynew[:,k,r] = (aux==k).astype(np.int)
      else:
        Ynew[:,k,r] = (aux==k+1).astype(np.int)

  return Ynew

from sklearn.base import  BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.model_selection import StratifiedShuffleSplit


  # Define custom loss

def custom_loss2(K,R):
  #pi [N,R]

  #@tf.function()  #decorador para operar sobre python, mas lento y poco efectivo en muchos casos
  # Create a loss function that adds the MSE loss to the mean of all squared activations of a specific layer
  def max_like(y_true, y_pred): # y_true [batch_size, K, R]
    #kernels###############################################
    N = y_true.shape[0]
    #K = y_pred[:, R:].shape[1]
    #K = np.unique(y_true).size
    y = y_pred[:, 0:1]
    y_hat = tf.repeat(y, R, axis = -1)
    #print(y_hat.numpy())
    Vnr = y_pred[:, 1:]
    

   #temp1 = Z*p_logreg           
    #temp2 = (1-Z)/K
      
    #Likelihood 

    temp1 = tf.math.log(Vnr) 

    temp2 = tf.math.square(tf.math.add(y_true,-y_hat))
    temp3 = tf.math.divide(temp2,Vnr)

    #####funcion de costo############################################
    f = 0.5*tf.math.reduce_sum(tf.math.add(temp1,temp3))
    return f
  
  # Return a function
  return max_like

def scheduler1(step = 10, ratio = 1.2):
  def scheduler(epoch, lr):
    if epoch % step == 0 and epoch>1:
      return lr/ratio
    else:
      return lr
  return scheduler


def scheduler2(epoch, lr):
  if epoch < 10:
    return lr
  else:
    return lr * tf.math.exp(-0.01)


class Keras_MA_pi_kern(BaseEstimator, TransformerMixin):
  def __init__(self, K, R, P, w_init='PCA', epochs=30,batch_size=64,learning_rate=1e-3,optimizer='RMS',
               validation_split=0.3, verbose=1, ratio=1.2, scale=0.1, seed=None, ratio_lin=0.75,
               l1=0.01, l2=0.01, RBFout=100, dropout=True, BN=True):
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
        self.w_init = w_init
        self.seed = seed
        self.RBFout = RBFout
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
    y = X[:,self.P:]
    Xt = X[:,:self.P]

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

    inputA = tf.keras.layers.Input(shape=(Xt.shape[1]), name='entradaA')
    if self.BN:
      inputB = tf.keras.layers.BatchNormalization()(inputA)
    else:
      inputB = inputA
    if self.dropout:
      inputB = tf.keras.layers.AlphaDropout(rate=0.2)(inputB)
    #if self.dropout:
    #  inputA = tf.keras.layers.AlphaDropout(rate=0.2)(inputA)

    Q1 = 1 ## num clases
    Q2 = self.R ## num anotadores
    Q3 = int(self.ratio_lin*self.P)#self.ratio_lin

    initializer = tf.keras.initializers.GlorotNormal(seed=self.seed)

    if self.w_init == 'PCA':
      pca = PCA().fit(Xt).components_
      initializer1 = tf.keras.initializers.Constant(value=pca[:Q3,:].T)
    elif self.w_init == 'inv':
      enc = OneHotEncoder()
      ymv = mode(y2,1)[0]
      yb = enc.fit_transform(ymv).toarray()
      Q3 = self.K
      w_init = np.linalg.pinv(Xt).dot(yb)
      initializer1 = tf.keras.initializers.Constant(value=w_init)
    else:
      initializer1 = initializer

    hW1 = tf.keras.layers.Dense(Q3,activation='relu',name='W',  bias_initializer='zeros', kernel_initializer=initializer1,
                              kernel_regularizer=tf.keras.regularizers.l1_l2(l1=self.l1,l2=self.l2))(inputB)
    if self.BN:
      hW1 = tf.keras.layers.BatchNormalization()(hW1)
    if self.dropout:
      hW1 = tf.keras.layers.AlphaDropout(rate=0.2)(hW1)

    hW1_1 = tf.keras.layers.Dense(self.RBFout,activation='relu',name='W2',  bias_initializer='zeros', kernel_initializer=initializer,
                              kernel_regularizer=tf.keras.regularizers.l1_l2(l1=self.l1,l2=self.l2))(hW1)                         
    if self.BN:
      hW1_1 = tf.keras.layers.BatchNormalization()(hW1_1)
    if self.dropout:
      hW1_1 = tf.keras.layers.AlphaDropout(rate=0.2)(hW1_1)
    #hW1_1 = tf.keras.layers.experimental.RandomFourierFeatures(output_dim=self.RBFout, name='RBF', kernel_initializer='gaussian', scale=self.scale)(hW1)
    
    hW = tf.keras.layers.Dense(Q1,activation='linear',name='Yhat',  bias_initializer='zeros', kernel_initializer=initializer,
                              kernel_regularizer=tf.keras.regularizers.l1_l2(l1=self.l1,l2=self.l2))(hW1_1)
    

    hW2 = tf.keras.layers.Dense(Q2,activation='exponential',name='PI', bias_initializer='zeros', kernel_initializer=initializer,
                                kernel_regularizer=tf.keras.regularizers.l1_l2(l1=self.l1,l2=self.l2))(hW1_1)

    concAB = tf.keras.layers.concatenate([hW,hW2],name='concAB')

    self.model = tf.keras.Model(inputs=inputA, outputs=concAB)
    self.model.compile(loss=custom_loss2(K=self.K, R= self.R), 
              optimizer=opt) #f1, precision, recall, crossentropy
            

    callback1 = tf.keras.callbacks.TerminateOnNaN()
    callback2 = tf.keras.callbacks.LearningRateScheduler(scheduler1(ratio = self.ratio))
    #callback2 = tf.keras.callbacks.LearningRateScheduler(scheduler2)
    callback3 = tf.keras.callbacks.EarlyStopping(monitor="loss", min_delta=1e-2,
                                                 patience=15, verbose=0, mode="auto",
                                                 baseline=None, restore_best_weights=True)

    self.history = self.model.fit(x = Xt, y = y,
                        epochs=self.epochs,batch_size=self.batch_size,
                        validation_split=self.validation_split, 
                        callbacks = [callback1, callback2],# callback3], 
                        verbose = self.verbose)
  
  def predict(self, X, *_):
    pred = self.model.predict(X[:,:self.P])[:,:self.K]
    return pred

  def plot_history_loss(self):
      plt.plot(self.history.history['loss'],label='loss')
      #plt.plot(self.history.history['val_loss'],label='val_loss')
      plt.legend()
      return

  def score(self, X, t=None):
    y_pred = self.model.predict(X[:,:self.P])
    y = X[:,self.P:]
    
    N = y.shape[0]
    #K = y_pred[:, R:].shape[1]
    #K = np.unique(y_true).size
    y_hat = np.repeat(y_pred[:,:1], self.R, axis = -1)

    Vnr = y_pred[:, 1:]
      
    #Likelihood 

    temp1 = np.log(Vnr) 

    temp2 = np.square(y-y_hat)
    temp3 = temp2/Vnr  

    #####funcion de costo############################################
    f = -0.5*np.sum(temp1+temp3)
    
    return f


