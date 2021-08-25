from scipy.special import softmax
import numpy as np 
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 
import matplotlib.pyplot as plt 
import random
from IPython.display import clear_output
from sklearn.base import BaseEstimator, ClassifierMixin
import pandas as pd 
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import OneHotEncoder 
from scipy.special import softmax
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_auc_score
from scipy.stats import mode 
from numpy.linalg import norm

def lr_sch(it, lr, step = 10, ratio = 2):
  if it % step == 0:
    return lr/ratio
  else:
    return lr

def multiclass_roc_auc_score(y_test, y_pred, average="macro"):
  lb = LabelBinarizer()
  lb.fit(y_test)
  y_test = lb.transform(y_test)
  y_pred = lb.transform(y_pred)
  return roc_auc_score(y_test, y_pred, average=average)

def ook_tf(Y, K):
    """Transforms Y to 1 of k notation
    n --> data points
    K --> classes
    Y_1ok --> nxK
    """
    
    Y_1ok = tf.one_hot(tf.dtypes.cast(Y, tf.int32),K)

    return Y_1ok


def z_post_tf_pilin(pi,p_logreg,K):
    """Calculates the posterior distribution of the latent variable z
    P --> data dimension
    R --> annotators
    N --> data points
    K --> classes
    X --> Pxn
    w_old --> PxK
    p_logreg --> NxR
    pi_old --> Rx1
    post --> RxN
    """
    N = p_logreg.shape[0]

    piM = 1-pi

    temp1 = pi*p_logreg           
    temp2 = piM*K
    post = temp1/tf.math.add(temp1,temp2)
    return post

def z_post_tf(pi,p_logreg,K):
    """Calculates the posterior distribution of the latent variable z
    P --> data dimension
    R --> annotators
    N --> data points
    K --> classes
    X --> Pxn
    w_old --> PxK
    p_logreg --> NxR
    pi_old --> Rx1
    post --> RxN
    """
    N = p_logreg.shape[0]
    piM = tf.tile(tf.transpose(pi), [N, 1])
    piM1 = tf.tile((1-tf.transpose(pi)), [N, 1])

    temp1 = piM*p_logreg           
    temp2 = piM1*K
    post = temp1/tf.math.add(temp1,temp2)
    return post

def np2tf(arg):
  arg = tf.convert_to_tensor(arg, dtype=tf.float32)
  return arg

def p_logreg_tf(w,X,Y,R,K):
    """Calculates the likelihood of the label provided by the rth annotator
    according to a k-class logreg model with parameters w
    p --> data dimension
    n --> data points
    K --> classes
    X --> pxn
    w --> pxK
    p_lr --> NxR
    """
    y_lr = tf.nn.softmax(tf.tensordot(X,w,1))


    t_nk = []

    for r in range(R):
        t_nk.append(ook_tf(Y[:,r],K)) 


    p_lr = []

    for r in range(R):
        p_lr.append(tf.math.reduce_prod(tf.math.pow((y_lr), t_nk[r]), axis=1))

    return tf.transpose(p_lr)

def p_rand_tf(K):
    return tf.constant(1/K)

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

def data_partition(X,Y,ind,frac,seed):
  import math
  """ Function to partition data into train, test
  ind  --> matrix with shufled index (w,N) where w are the different possible random states
  frac --> fraction of data for training
  seed --> fixed random state for reproducibility
  """
  N = X.shape[0]
  Xtrain = X[ind[seed][:math.ceil(N*frac)]]
  Ytrain = Y[ind[seed][:math.ceil(N*frac)]]

  Xtest = X[ind[seed][math.ceil(N*frac):]]
  Ytest = Y[ind[seed][math.ceil(N*frac):]]

  return Xtrain, Ytrain, Xtest, Ytest

def med_dist(X):
  from sklearn.metrics import pairwise_distances
  return np.median(pairwise_distances(X, metric='euclidean'))


class MA_pi_linear_minibatch(BaseEstimator, ClassifierMixin):
 
  def __init__(self, max_iter = 500, learning_rate = 0.01, tol = 0.0001, plots = False, batch_size = 150):
    self.max_iter = max_iter
    self.learning_rate = learning_rate
    self.tol = tol
    self.plots = plots
    self.batch_size = batch_size

  def fit(self, X, Y,t):

    X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
    R = Y.shape[1] 
    N = self.batch_size
    P = X.shape[1]
    K = np.unique(t).size
    N_tf = tf.constant(1/N,dtype=tf.float32)
    K_tf = tf.constant(1/K,dtype=tf.float32)
    ind = np.arange(X.shape[0])

    # Model inputs
    X_tf = tf.placeholder(tf.float32, [N, X.shape[1]])
    Y_tf = tf.placeholder(tf.float32, [N, Y.shape[1]])

    # Variables
    enc = OneHotEncoder()
    ymv = mode(Y,1)[0]
    yb = enc.fit_transform(ymv).toarray()
    w_init = np2tf(np.linalg.pinv(X).dot(yb))
    w = tf.Variable(w_init)
    #w = tf.Variable(tf.zeros([P, K], tf.float32))
    #w2 = tf.Variable(0.01*tf.ones([P, R], tf.float32))
    w2_init = np2tf(np.linalg.pinv(X).dot((Y==ymv).astype(int)))
    w2 = tf.Variable(w2_init)
    learning_rate = tf.placeholder(tf.float32, shape=[])
    #pi = tf.Variable(0.1*tf.ones([N, R], tf.float32), trainable=False)
    Z_p = tf.Variable(tf.ones([N, Y.shape[1]], tf.float32), trainable=False)

    # Useful variables
    Yhat = tf.nn.softmax(tf.tensordot(X_tf,w,1))
    p_logreg = p_logreg_tf(w,X_tf,Y_tf,R,K)
    pi = tf.nn.sigmoid(tf.tensordot(X_tf,w2,1)) 

    # Expectation step
    auxZ_p = z_post_tf_pilin(pi,p_logreg,K_tf)
    assingZ = tf.assign(Z_p, auxZ_p)
    Z_p_com = 1-Z_p

    # Maximization step

    #assingpi = tf.assign(pi, auxpi)
    # Cost function for w (we also could use L; we would obtain the same parameters since pi and Z_p are non-trainable)
    C = Z_p*tf.math.log(pi*p_logreg)     

    #Likelihood 
    temp1 = C         
    temp2 = Z_p_com*tf.math.log(pi*K_tf)   
    L = tf.math.reduce_sum(tf.math.add(temp1,temp2))

    #optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(-L)
    #optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate,beta1=0.9,beta2=0.9,epsilon=1e-08).minimize(-L)
    lr = self.learning_rate
    optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(-L)
    Lik_acum=[]
    acc_acum=[]
    W_acum = []
    W2_acum = []
    auc_acum = []
    lb = LabelBinarizer()

    total_batch = int(N/self.batch_size)


    if len(np.unique(Y))!=2:
      y = lb.fit_transform(t)
    else:
      y = lb.fit_transform(t)
      y = np.concatenate((1-y.astype(bool), y), axis = 1)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        for it in range(self.max_iter):
          avg_Lik = 0
          avg_W_norm = 0
          avg_W2_norm = 0
          random.shuffle(ind)

          for i in range(total_batch):

            idx = ind[i*self.batch_size:((i+1)*self.batch_size)]
            batch_xs, batch_ys, batch_ts = X[idx,:], Y[idx,:], t[idx,:]

            ### programar paso E
            sess.run(assingZ, feed_dict={X_tf: batch_xs, Y_tf: batch_ys})

            ## Paso M
            lr = lr_sch(it, lr, step = 10, ratio = 1.5)
            sess.run(optimizer, feed_dict={X_tf: batch_xs, Y_tf: batch_ys, learning_rate: lr})

            #Analyze the likelihood 
            Lik = sess.run(L, feed_dict={X_tf: batch_xs, Y_tf: batch_ys})
            avg_Lik += Lik/total_batch

            Yhat1 = sess.run(Yhat, feed_dict={X_tf: batch_xs, Y_tf: batch_ys})

            avg_W_norm += norm(sess.run(w))/total_batch
            avg_W2_norm += norm(sess.run(w2))/total_batch
            #print(Lik)
          
          Yhat2 = softmax(np.dot(X,sess.run(w)))
          pred = np.argmax(Yhat2, axis=1)
          correct_prediction = np.equal(pred, t.T) 
          acc = np.mean(correct_prediction)
          acc_acum.append(acc)

          Lik_acum.append(avg_Lik)
          W_acum.append(avg_W_norm) 
          W2_acum.append(avg_W2_norm)

          auc_acum.append(roc_auc_score(y, Yhat2))
        # Let's train the regressor
        Weight = sess.run(w) # Optimized Weight  
        Weight2 = sess.run(w2)
        #print(sess.run(pi))
        #print(Weight2)
        
        #print(sess.run(Z_p))
        
        # Let's check how is performing the regressor
        #correct_prediction = tf.equal(tf.argmax(Yhat1, 1), (t).T) 
        #accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        #print(accuracy.eval())
        #print(Yhat1)
        #print(tf.argmax(Yhat1, 1).eval())
    if self.plots:
      plt.plot(Lik_acum)
      plt.title('Cost function')
      plt.figure()
      plt.plot(acc_acum)
      plt.title('Accuracy')
      plt.figure()
      plt.plot(auc_acum)
      plt.title('AUC')
      plt.figure()
      plt.plot(W_acum)
      plt.title('W Norm')
      plt.figure()
      plt.plot(W2_acum)
      plt.title('W2 Norm')

    # Store the classes seen during fit
    self.X_ = X
    self.y_ = Y
    self.W_ = Weight
    self.W2_ = Weight2
    # Return the classifier
    return self


  def predict(self, X):
    X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
    Yhat = softmax(np.dot(X,self.W_))
    self.prediction = Yhat
    return self.prediction


