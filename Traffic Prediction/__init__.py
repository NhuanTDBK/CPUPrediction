import numpy as np
import pandas as pd
import matplotlib.pyplot as pl
import sklearn
from sknn.mlp import Regressor, Layer
import pickle
from sknn.platform import cpu64
raw_data = pd.read_csv("ita_public_tools/output/data.csv");
def get_training():
 #   n_row = 578 
 # group du lieu
    data = raw_data.groupby('Timestamp').count()["Timestamp"]
    n_input = 10
    n_row = data.shape[0]
    print "Generate X_traing, y_traing"
    print "X_training loading..."
    X_training = np.asarray([[data[t-i-1] for i in range(0,n_input)]
                 for t in np.arange(n_input,n_row)])
    print "y_training loading..."
    y_training = np.asarray(data[n_input:n_row])
    n_sample2 = np.asarray([[data[t-i-1] for i in range(0,n_input)] for t in np.arange (289*400,289*410)])
    print "y_test..."
    n_test2 =  np.asarray([raw_data.ix[t][4] for t in np.arange(289*400+1,289*410+1)])
    return X_training, y_training,n_sample2,n_test2
