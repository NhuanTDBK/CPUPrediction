import numpy as np
import pandas as pd
import matplotlib.pyplot as pl
import sklearn
from sknn.mlp import Regressor, Layer
import pickle
from sknn.platform import cpu64
raw_data = pd.read_csv("tpds-2012-workload.csv");
def get_training():
#     n_row = 578 
    n_row = raw_data.shape[0]
    n_input = 10
    data = raw_data["Utilization"]
    print "Generate X_traing, y_traing"
    print "X_training loading..."
    X_training = np.asarray([[data[t-i-1] for i in range(0,n_input)]
                 for t in np.arange(n_input,n_row)])
    print "y_training loading..."
    y_training = np.asarray(data[n_input:n_row])
    return X_training, y_training
