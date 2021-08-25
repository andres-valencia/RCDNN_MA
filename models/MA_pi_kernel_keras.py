
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
  lb = LabelBinarizer()
  lb.fit(np.unique(Y))
  N = Y.shape[0] 
  K = len(np.unique(Y))
  R = Y.shape[1]
  Ynew = np.zeros([N,K,R])

  for i in range(N):
    y_b = lb.transform(Y[i])

    if K == 2:
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
    temp2 = (1-pi)*K
    Zp = temp1/tf.math.add(temp1,temp2)
      
    #Likelihood 
    C = Zp*tf.math.log(pi*p_logreg)   
    temp1 = C         
    temp2 = (1-Zp)*tf.math.log(pi*1/K)   

    #####funcion de costo############################################
    f = -tf.math.reduce_sum(tf.math.add(temp1,temp2))
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
               l1=0.01, l2=0.01, RBFout=100):
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate=learning_rate 
        self.validation_split = validation_split
        self.verbose = verbose
        self.optimizer = optimizer
        self.ratio = ratio
        self.K = K
        self.R = R
        self.P = P
        self.scale = scale
        self.l1 = l1
        self.l2 = l2
        self.w_init = w_init
        self.seed = seed
        self.RBFout = RBFout
        self.ratio_lin = ratio_lin

        
  def fit(self, X, t):
    #lb = LabelBinarizer()
    #lb.fit(X[:,-1])
    #N = X.shape[0]
    #y = np.zeros([N, self.K, self.R])
    #for i in range(N):
    #  y[i,:,:] = binarize(X[i,self.P:],lb).T
    y2 = X[:,self.P:]
    y = bin_Y(y2)

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
    Q3 = int(self.ratio_lin*self.P)

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

    hW1 = tf.keras.layers.Dense(Q3,activation='linear',name='W',  bias_initializer='zeros', kernel_initializer=initializer1,
                              kernel_regularizer=tf.keras.regularizers.l1_l2(l1=self.l1,l2=self.l2))(inputA)
    
    hW1_1 = tf.keras.layers.experimental.RandomFourierFeatures(output_dim=self.RBFout, name='RBF', kernel_initializer='gaussian', scale=self.scale)(hW1)
    
    hW = tf.keras.layers.Dense(Q1,activation='softmax',name='Yhat',  bias_initializer='zeros', kernel_initializer=initializer,
                              kernel_regularizer=tf.keras.regularizers.l1_l2(l1=self.l1,l2=self.l2))(hW1_1)
    
    hW2 = tf.keras.layers.Dense(Q2,activation='sigmoid',name='PI', bias_initializer='zeros', kernel_initializer=initializer,
                                kernel_regularizer=tf.keras.regularizers.l1_l2(l1=self.l1,l2=self.l2))(hW1_1)

    concAB = tf.keras.layers.concatenate([hW,hW2],name='concAB')

    self.model = tf.keras.Model(inputs=inputA, outputs=concAB)
    self.model.compile(loss=custom_loss(K=self.K, R= self.R), 
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
    return np.argmax(pred, axis=1)
  
  def predict_proba(self, X, *_):
    pred = self.model.predict(X[:,:self.P])[:,:self.K]
    return pred

  def plot_history_loss(self):
      plt.plot(self.history.history['loss'],label='loss')
      #plt.plot(self.history.history['val_loss'],label='val_loss')
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
