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

def ook_tf(Y):
    """Transforms Y to 1 of k notation
    n --> data points
    K --> classes
    Y_1ok --> nxK
    """
    
    Y_1ok = tf.one_hot(tf.dtypes.cast(Y, tf.int32),3)

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

def p_logreg_tf(w,X,Y,R):
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
        t_nk.append(ook_tf(Y[:,r])) 


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

class MA_linear(BaseEstimator, ClassifierMixin):
 
  def __init__(self, max_iter = 200, learning_rate = 0.01, tol = 0.0001):
    self.max_iter = max_iter
    self.learning_rate = learning_rate
    self.tol = tol

  def fit(self, X, Y,t):

    X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)

    R = Y.shape[1] 
    N = X.shape[0]
    K = np.unique(t).size
    N_tf = tf.constant(1/N,dtype=tf.float32)
    K_tf = tf.constant(1/K,dtype=tf.float32)

    # Model inputs
    X_tf = tf.placeholder(tf.float32, [X.shape[0], X.shape[1]])
    Y_tf = tf.placeholder(tf.float32, [X.shape[0], Y.shape[1]])

    # Variables
    w = tf.Variable(tf.zeros([X.shape[1], K], tf.float32))
    pi = tf.Variable(0.1*tf.ones([Y.shape[1], 1], tf.float32), trainable=False)
    Z_p = tf.Variable(0.5*tf.ones(Y.shape, tf.float32), trainable=False)

    # Useful variables
    Yhat = tf.nn.softmax(tf.tensordot(X_tf,w,1))
    p_logreg = p_logreg_tf(w,X_tf,Y_tf,R)

    # Expectation step
    auxZ_p = z_post_tf(pi,p_logreg,K_tf)
    assingZ = tf.assign(Z_p, auxZ_p)
    Z_p_com = 1-Z_p

    # Maximization step
    auxpi = tf.transpose(tf.math.multiply(tf.reduce_sum(Z_p, axis=0, keepdims=True),N_tf))
    assingpi = tf.assign(pi, auxpi)
    # Cost function for w (we also could use L; we would obtain the same parameters since pi and Z_p are non-trainable)
    C = Z_p*tf.math.log(tf.tile(tf.transpose(pi), [N, 1])*p_logreg)     

    #Likelihood 
    temp1 = C         
    temp2 = Z_p_com*tf.math.log(tf.tile(tf.transpose(1-pi), [N, 1])*K_tf)   
    L = tf.math.reduce_sum(tf.math.add(temp1,temp2))

    optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(-C)

    #optimizer = tf.train.AdamOptimizer(learning_rate=0.01,beta1=0.9,beta2=0.9,epsilon=1e-08).minimize(-temp1)
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        for it in range(self.max_iter):
            ### programar paso E
            sess.run(assingZ, feed_dict={X_tf: X, Y_tf: Y})

            ## Paso M
            sess.run([optimizer,assingpi], feed_dict={X_tf: X, Y_tf: Y})

            #Analyze the likelihood 
            Lik = sess.run(L, feed_dict={X_tf: X, Y_tf: Y})
            Yhat1 = sess.run(Yhat, feed_dict={X_tf: X, Y_tf: Y})
            #print(Lik)

        # Let's train the regressor
        Weight = sess.run(w) # Optimized Weight  
        PI = sess.run(pi)
        print(PI)
    
    #print(sess.run(Z_p))
    
    # Let's check how is performing the regressor
     
    #print(Yhat1)
    #print(tf.argmax(Yhat1, 1).eval())
    # Check that X and y have correct shape 
    # Store the classes seen during fit
    self.X_ = X
    self.y_ = Y
    self.W_ = Weight
    self.pi_ = PI
    # Return the classifier
    return self

  def predict(self, X):
    X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
    Yhat = softmax(np.dot(X,self.W_))
    self.prediction = np.argmax(Yhat, axis=1)
    return self.prediction


class Logreg_cla(BaseEstimator, ClassifierMixin):

  
  def __init__(self, alpha=0.0025, epochs=100, batch_size=15, plots = False ):
    self.alpha = alpha
    self.epochs = epochs
    self.batch_size = batch_size
    self.plots = plots


  def fit(self, x, t):
    # Creating the One Hot Encoder 
    oneHot = OneHotEncoder() 
    m, n = x.shape 
    x = np.concatenate([x,np.reshape(np.ones(m),[m,1])],1)

    # Encoding y_orig 
    t=t.reshape(-1, 1)
    oneHot.fit(t)
    y = oneHot.transform(t).toarray() 
    ind = np.arange(m)

    #print('P =', m) 
    #print('N =', n) 
    #print('Learning Rate =', self.alpha) 
    #print('Number of Epochs =', self.epochs) 
    #print('Batch Size =', self.batch_size) 

    X = tf.placeholder(tf.float32, [None, n+1]) 
 
    Y = tf.placeholder(tf.float32, [None, np.unique(t).size]) 
      
    # Trainable Variable Weights 
    B = tf.Variable(tf.zeros([n+1,np.unique(t).size]))

    # Hypothesis 
    Y_hat = tf.nn.softmax(tf.matmul(X, B)) 
      
    # Sigmoid Cross Entropy Cost Function ##########Cambiar por softmax
    cost = tf.nn.softmax_cross_entropy_with_logits_v2( 
                        logits = Y_hat, labels = Y) 
      
    # Gradient Descent Optimizer 
    optimizer = tf.train.GradientDescentOptimizer( 
            learning_rate = self.alpha).minimize(cost) 
      
    # Global Variables Initializer 
    init = tf.global_variables_initializer()

    # Starting the Tensorflow Session 
    with tf.Session() as sess: 
          
      # Initializing the Variables 
      sess.run(init) 
        
      # Lists for storing the changing Cost and Accuracy in every Epoch 
      cost_history, accuracy_history = [], [] 
        
      # Iterating through all the epochs 
      for epoch in range(self.epochs): 
        #print(epoch)
        cost_per_epoch = 0
        random.shuffle(ind)     
        total_batch = int(m/self.batch_size)
        avg_cost = 0
        # Loop over all batches

        for i in range(total_batch):
          idx = ind[i*self.batch_size:((i+1)*self.batch_size)]
          batch_xs, batch_ys = x[idx,:], y[idx,:]
          # Run optimization op (backprop) and cost op (to get loss value)
          _, c = sess.run([optimizer, cost], feed_dict={X: batch_xs, Y: batch_ys})
          # Compute average loss
          avg_cost += c / total_batch
          
        # Calculating accuracy on current Epoch 
        correct_prediction = tf.equal(tf.argmax(Y_hat, 1), tf.argmax(Y, 1)) 
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) 
          
        # Storing Cost and Accuracy to the history 
        cost_history.append(np.sum(np.sum(avg_cost))) 
        accuracy_history.append(accuracy.eval({X : x, Y : y}) * 100) 
          
        # Displaying result on current Epoch 
        #if epoch % 100 == 0 and epoch != 0: 
          # print("Epoch " + str(epoch) + " Avg Cost: "
          #                  + str(cost_history[-1])) 
      
      Weight = sess.run(B) # Optimized Weight  
      
      # Final Accuracy 
      correct_prediction = tf.equal(tf.argmax(Y_hat, 1), tf.argmax(Y, 1)) 
      accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) 
      #print(accuracy)
      # print("\nAccuracy:", accuracy_history[-1], "%") 

    
    # Store the classes seen during fit
    self.X_ = x
    self.y_ = y
    self.W_ = Weight

    if self.plots:
      plt.plot(accuracy_history)
      plt.figure()
      plt.plot(cost_history)


    # Return the classifier
    return self

  
  def predict(self, X):
    m,n = X.shape 
    X = np.concatenate([X,np.reshape(np.ones(m),[m,1])],1)
    Yhat = softmax(np.dot(X,self.W_))
    self.prediction = np.argmax(Yhat, axis=1)
    return self.prediction 


class MA_pi_linear(BaseEstimator, ClassifierMixin):
 
  def __init__(self, max_iter = 200, learning_rate = 0.01, tol = 0.0001):
    self.max_iter = max_iter
    self.learning_rate = learning_rate
    self.tol = tol

  def fit(self, X, Y,t):

    X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
    R = Y.shape[1] 
    N = X.shape[0]
    P = X.shape[1]
    K = np.unique(t).size
    N_tf = tf.constant(1/N,dtype=tf.float32)
    K_tf = tf.constant(1/K,dtype=tf.float32)

    # Model inputs
    X_tf = tf.placeholder(tf.float32, [X.shape[0], X.shape[1]])
    Y_tf = tf.placeholder(tf.float32, [X.shape[0], Y.shape[1]])

    # Variables
    w = tf.Variable(tf.zeros([P, K], tf.float32))
    w2 = tf.Variable(tf.zeros([P, R], tf.float32))
    #pi = tf.Variable(0.1*tf.ones([N, R], tf.float32), trainable=False)
    Z_p = tf.Variable(0.5*tf.ones(Y.shape, tf.float32), trainable=False)

    # Useful variables
    Yhat = tf.nn.softmax(tf.tensordot(X_tf,w,1))
    p_logreg = p_logreg_tf(w,X_tf,Y_tf,R)
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

    optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(-L)

    #optimizer = tf.train.AdamOptimizer(learning_rate=0.01,beta1=0.9,beta2=0.9,epsilon=1e-08).minimize(-temp1)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        for it in range(self.max_iter):
            ### programar paso E
            sess.run(assingZ, feed_dict={X_tf: X, Y_tf: Y})

            ## Paso M
            sess.run([optimizer], feed_dict={X_tf: X, Y_tf: Y})

            #Analyze the likelihood 
            Lik = sess.run(L, feed_dict={X_tf: X, Y_tf: Y})
            Yhat1 = sess.run(Yhat, feed_dict={X_tf: X, Y_tf: Y})
            #print(Lik)

        # Let's train the regressor
        Weight = sess.run(w) # Optimized Weight  
        Weight2 = sess.run(w2)
        #print(sess.run(pi))
        #print(Weight2)
        
        #print(sess.run(Z_p))
        
        # Let's check how is performing the regressor
        correct_prediction = tf.equal(tf.argmax(Yhat1, 1), (t).T) 
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print(accuracy.eval())
        #print(Yhat1)
        #print(tf.argmax(Yhat1, 1).eval())

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
    self.prediction = np.argmax(Yhat, axis=1)
    return self.prediction



class MA_pi_kern(BaseEstimator, ClassifierMixin):
 
  def __init__(self, max_iter = 200, learning_rate = 0.01, tol = 0.0001):
    self.max_iter = max_iter
    self.learning_rate = learning_rate
    self.tol = tol

  def fit(self, X, Y,t):
    Xtrain = X
    gamma = 0.5/med_dist(X)**2 #1/med_dist(X)**2
    X = rbf_kernel(X, gamma=gamma)

    X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
    R = Y.shape[1] 
    N = X.shape[0]
    P = X.shape[1]
    K = np.unique(t).size
    N_tf = tf.constant(1/N,dtype=tf.float32)
    K_tf = tf.constant(1/K,dtype=tf.float32)

    # Model inputs
    X_tf = tf.placeholder(tf.float32, [X.shape[0], X.shape[1]])
    Y_tf = tf.placeholder(tf.float32, [X.shape[0], Y.shape[1]])

    # Variables
    w = tf.Variable(tf.zeros([P, K], tf.float32))
    w2 = tf.Variable(tf.zeros([P, R], tf.float32))
    #pi = tf.Variable(0.1*tf.ones([N, R], tf.float32), trainable=False)
    Z_p = tf.Variable(0.5*tf.ones(Y.shape, tf.float32), trainable=False)

    # Useful variables
    Yhat = tf.nn.softmax(tf.tensordot(X_tf,w,1))
    p_logreg = p_logreg_tf(w,X_tf,Y_tf,R)
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

    optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(-L)

    #optimizer = tf.train.AdamOptimizer(learning_rate=0.01,beta1=0.9,beta2=0.9,epsilon=1e-08).minimize(-temp1)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        for it in range(self.max_iter):
            ### programar paso E
            sess.run(assingZ, feed_dict={X_tf: X, Y_tf: Y})

            ## Paso M
            sess.run([optimizer], feed_dict={X_tf: X, Y_tf: Y})

            #Analyze the likelihood 
            Lik = sess.run(L, feed_dict={X_tf: X, Y_tf: Y})
            Yhat1 = sess.run(Yhat, feed_dict={X_tf: X, Y_tf: Y})
            #print(Lik)

        # Let's train the regressor
        Weight = sess.run(w) # Optimized Weight  
        Weight2 = sess.run(w2)
        #print(sess.run(pi))
        #print(Weight2)
        
        #print(sess.run(Z_p))
        
        # Let's check how is performing the regressor
        correct_prediction = tf.equal(tf.argmax(Yhat1, 1), (t).T) 
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print(accuracy.eval())
        #print(Yhat1)
        #print(tf.argmax(Yhat1, 1).eval())

    # Store the classes seen during fit
    self.X_ = X
    self.Xtrain_ = Xtrain
    self.y_ = Y
    self.t_ = t
    self.W_ = Weight
    self.W2_ = Weight2
    self.gamma_ = gamma
    # Return the classifier
    return self


  def predict(self, X):
    X = rbf_kernel(X, self.Xtrain_, gamma=self.gamma_)
    X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
    Yhat = softmax(np.dot(X,self.W_))
    self.prediction = np.argmax(Yhat, axis=1)
    return self.prediction
# Generate shuffled index matrix for reproducibility

#ind = np.arange(150)
#sh_ind = []
#random.shuffle(ind) 
#for i in range(100):
#  sh_ind.append(ind)
#  ind = np.arange(150)
#  random.shuffle(ind)    
#from scipy.io import savemat
#index = {"sh_ind": sh_ind}
#savemat("data/shuffled_index.mat", index)

#sh_ind = loadmat('data/shuffled_index.mat')['sh_ind']
#sh_ind.shape

#sh_ind[1][50:]

