
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

  if len(np.unique(t))!=2:
    lb = LabelBinarizer()
    y_ook = lb.fit_transform(t)
  else:
    lb = LabelBinarizer()
    y_ook = lb.fit_transform(t)
    y_ook = np.concatenate((1-y_ook.astype(bool), y_ook), axis = 1) 

  return y_ook


def bin_Y(Y):
  lb = LabelBinarizer()
  lb.fit(np.unique(Y))
  N = Y.shape[0] 
  K = len(np.unique(Y))
  R = Y.shape[1]
  Ynew = np.zeros([N,K,R])
  for i in range(N):

    if K != 2:
      y_b = lb.transform(Y[i])

    else:
      y_b = lb.transform(Y[i])
      y_b = np.concatenate((1-y_b.astype(bool), y_b), axis = 1) 

    Ynew[i,:,:] = y_b.T

  return Ynew

from sklearn.base import  BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.model_selection import StratifiedShuffleSplit


  # Define custom loss
def custom_loss(K,R):
  #pi [N,R]

  #@tf.function()  #decorador para operar sobre python, mas lento y poco efectivo en muchos casos
  # Create a loss function that adds the MSE loss to the mean of all squared activations of a specific layer
  def max_like(y_true, y_pred): # y_true [batch_size, K, R]
    #kernels###############################################
    N = y_true.shape[0]
    #K = y_pred[:, R:].shape[1]
    #K = np.unique(y_true).size
    y_hat = tf.repeat(tf.expand_dims(y_pred[:, :K],-1), R, axis = -1)

    pi = y_pred[:, K:]

    p_logreg = tf.math.reduce_prod(tf.math.pow(y_hat, y_true), axis=1)

    temp1 = pi*p_logreg           
    temp2 = (1-pi)/K
    Zp = temp1/tf.math.add(temp1,temp2)
      
    #Likelihood 

    temp1 = Zp*tf.math.log(pi*p_logreg)       
    temp2 = (1-Zp)*tf.math.log((1-pi)*1/K)   

    #####funcion de costo############################################
    f = -tf.math.reduce_sum(tf.math.add(temp1,temp2))
    return f
  
  # Return a function
  return max_like

def scheduler1(step = 8, ratio = 1.2):
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


class Keras_MA_pi_lin(BaseEstimator, TransformerMixin):
  def __init__(self, K, R, P, epochs=30,batch_size=64,learning_rate=1e-3,optimizer='RMS',
               l1_param=1e-3,l2_param=1e-3,validation_split=0.3,verbose=1,
               w_init=False, w2_init=False, ratio = 1.2):
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate=learning_rate 
        self.l1_param=l1_param 
        self.l2_param=l2_param
        self.validation_split = validation_split
        self.verbose = verbose
        self.optimizer = optimizer
        self.w_init = w_init
        self.w2_init = w2_init
        self.ratio = ratio
        self.K = K
        self.R = R
        self.P = P

        
  def fit(self, X, t):
    #lb = LabelBinarizer()
    #lb.fit(X[:,-1])
    #N = X.shape[0]
    #y = np.zeros([N, self.K, self.R])
    #for i in range(N):
    #  y[i,:,:] = binarize(X[i,self.P:],lb).T

    y = bin_Y(X[:,self.P:])

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

    Q1 = self.K ## num clases
    Q2 = self.R ## num anotadores
    l1 = self.l1_param
    l2 = self.l2_param
    
    if self.w_init.any():
      initializer = tf.keras.initializers.Constant(value=self.w_init)
    else:
      initializer = tf.random_uniform_initializer(minval=-1, maxval=1)

    if self.w2_init.any():
      initializer2 = tf.keras.initializers.Constant(value=self.w2_init)
    else:
      initializer2 = tf.random_uniform_initializer(minval=-1, maxval=1)

    hW = tf.keras.layers.Dense(Q1,activation='softmax',name='Yhat', kernel_initializer=initializer, bias_initializer='zeros',
                              kernel_regularizer=tf.keras.regularizers.l1_l2(l1=l1,l2=l2))(inputA)
    
    hW2 = tf.keras.layers.Dense(Q2,activation='sigmoid',name='PI', kernel_initializer=initializer2, bias_initializer='zeros',
                                kernel_regularizer=tf.keras.regularizers.l1_l2(l1=l1,l2=l2))(inputA)

    concAB = tf.keras.layers.concatenate([hW,hW2],name='concAB')

    self.model = tf.keras.Model(inputs=inputA, outputs=concAB)
    self.model.compile(loss=custom_loss(K=self.K, R= self.R), 
              optimizer=opt) #f1, precision, recall, crossentropy

    callback1 = tf.keras.callbacks.TerminateOnNaN()
    callback2 = tf.keras.callbacks.LearningRateScheduler(scheduler1(ratio = self.ratio))
    #callback2 = tf.keras.callbacks.LearningRateScheduler(scheduler2)
    callback3 = tf.keras.callbacks.EarlyStopping(monitor="loss", min_delta=1e-3,
                                                 patience=10, verbose=0, mode="auto",
                                                 baseline=None, restore_best_weights=True)

    
    self.history = self.model.fit(x = Xt, y = y,
                        epochs=self.epochs,batch_size=self.batch_size,
                        validation_split=self.validation_split, 
                        callbacks = [callback1, callback2, callback3], 
                        verbose = self.verbose)
    
  def predict(self, X, *_):
    pred = self.model.predict(X[:,:self.P])[:,:self.K]
    return np.argmax(pred, axis=1)
  
  def predict_proba(self, X, *_):
    pred = self.model.predict(X[:,:self.P])[:,:self.K]
    return pred

  def plot_history_loss(self):
      plt.plot(self.history.history['loss'],label='loss')
      plt.plot(self.history.history['val_loss'],label='val_loss')
      plt.legend()
      return
  
  def score(self, X, t):
    accuracy = np.mean(self.predict(X)==t.T)

    try:
      auc = roc_auc_score(ook(t), self.predict_proba(X))
    except:
      auc = 0.5
    
    dist = np.sqrt((np.mean(accuracy)-1)**2+(np.mean(auc)-1)**2)

    return -dist
