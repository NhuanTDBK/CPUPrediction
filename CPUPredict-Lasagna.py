

import numpy as np
import pandas as pd
import matplotlib.pyplot as pl
import pickle
import sklearn
from sklearn.metrics import mean_squared_error
import theano
import lasagne as ls
from theano import tensor as T
from lasagne.layers import InputLayer, DenseLayer
from lasagne.updates import nesterov_momentum
from lasagne.nonlinearities import rectify
from nolearn.lasagne import NeuralNet

#Init data training
print "Reading file......"
raw_data = pd.read_csv("tpds-2012-workload.csv");
n_row = raw_data.shape[0]
n_input = 11
data = raw_data["Utilization"]
print "Generate X_training, y_training"
print "X_training loading..."
X_training = np.asarray([[data[i] for i in range(1,n_input)]
             for t in np.arange(n_input-1,n_row-1)])
print "y_training loading..."
y_training = data[n_input:]
if(X_training.shape[0]!=y_training.shape[0]):
    print "X_training shape must match y_training shape"
print "Multi Layer Perceptron..."
#Build layer for MLP
l_in = ls.layers.InputLayer(shape=(None,10),input_var=None)
l_hidden = ls.layers.DenseLayer(l_in,num_units=10,nonlinearity=ls.nonlinearities.rectify)
network = l_out = ls.layers.DenseLayer(l_hidden,num_units=1)
print "Neural network initialize"
#Init Neural net
net1 = NeuralNet(
    layers=network,
    # optimization method:
    update=nesterov_momentum,
    update_learning_rate=0.01,
    update_momentum=0.9,
    regression=True,  # flag to indicate we're dealing with regression problem
    max_epochs=400,  # we want to train this many epochs
    verbose=1,
)
#
print "Training time!!!!!....."
net1.fit(X_training,y_training)
net1.save_params_to("saveNeuralNetwork.tdn")
