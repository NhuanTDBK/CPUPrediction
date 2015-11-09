
# coding: utf-8

# In[1]:

from __future__ import division
import os
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import theano
T = theano.tensor
from sklearn.metrics import mean_squared_error
import pickle as pkl
from math import sqrt
from __init__ import *


# In[2]:

params = [100, 0.3, 250,10,20]
X_training, y_training = get_training()
# init pop size, mut rate, number of generation, chromoesome length, winner per gen]
fan_in = fan_out = 10


# In[ ]:




# In[6]:

class GA:
    def __init__(self,dataset,params,fan_in,fan_out):
        self.raw_data = dataset
        self.params = params
        self.lim = np.sqrt(6)/np.sqrt((fan_in+fan_out))
    def initPop(self):
        #random pick 100*10 element
        self.curPop = np.random.choice(np.arange(-self.lim,self.lim,step=0.0001),
                                  size=(self.params[0],self.params[3]),
                                  replace=False)
        # init next pop
        self.nextPop = np.zeros((self.curPop.shape[0],self.curPop.shape[1]))
        # fitness values for rank
        self.fitVec = np.zeros((self.params[0],2))
    def selection(self,X_training,y_training):
        for i in range(self.params[2]):
            self.fitVec = np.array([np.array([x, 
                                    np.sum(self.costFunction(X_training,y_training,self.curPop[x].reshape(10,1)))]) 
                                    for x in range(self.params[0])])
            winners = np.zeros((self.params[4],self.params[3]))
            for n in range(len(winners)):
                selected = np.random.choice(range(len(self.fitVec)),self.params[4]/2,replace=False)
                wnr = np.argmin(self.fitVec[selected,1])
                winners[n] = self.curPop[int(self.fitVec[selected[wnr]][0])]
            self.nextPop[:len(winners)] = winners
            self.nextPop[len(winners):] = np.array([
                    np.array(np.random.permutation(np.repeat(winners[:,x],(self.params[0]-len(winners))/len(winners),axis=0))) 
                    for x in range(winners.shape[1])]).T
            self.nextPop = np.multiply(self.nextPop,np.matrix([np.float(np.random.normal(0,2,1) if random.random() < self.params[1] else 1)
                                                                 for x in range(self.nextPop.size)]).reshape(self.nextPop.shape))
            self.curPop = self.nextPop
        return self.curPop[np.argmin(self.fitVec[:1])]
    def fit(self,X_training,y_training):
        self.initPop()
        return self.selection(X_training,y_training)
    def sigmoid(self,x):
        return 1.0 / (1 + np.exp(-x))
    def costFunction(self,X,y,theta):
        m = float(len(X))
        hThetaX = np.array(self.sigmoid(np.dot(X,theta)))
        return np.sum(np.abs(y-hThetaX))


# In[7]:

ga = GA(raw_data,params,fan_in,fan_out)
result = 1
for ty in range(0,10000):
    theta_in = ga.fit(X_training,y_training)
    pred_y = ga.sigmoid(np.dot(X_training,theta_in.T))
    tmp = sqrt(mean_squared_error(y_training,pred_y))
    if(result>tmp): 
            result = tmp
            best_y = pred_y


# In[ ]:

# ax = plt.subplot()
# ax.set_color_cycle(['blue','red'])
# ax.plot(y_training,label="actual")
# ax.plot(pred_y,label="predict")
# ax.legend()
# plt.show()


# In[ ]:

theta = theta_in #0.06
print "Ty le du doan " + result
pkl.dump(theta,open('saveGenetic.p', 'wb'))
# t = pkl.load(open('saveGenetic.p', 'rb'))


# In[ ]:



