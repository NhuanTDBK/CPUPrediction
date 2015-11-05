# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import numpy as np
import pandas as pd
import matplotlib.pyplot as pl
import sklearn
from sknn.mlp import Regressor, Layer
import pickle
from sknn.platform import cpu64
from __init__ import *
# <codecell>
print "Reading file......"
nn = Regressor(
    layers=[
        Layer("Rectifier", units=10),
        Layer("Linear")],
    learning_rate=0.02,
    n_iter=30)
X_training, y_training = get_training()
if(X_training.shape[0]!=y_training.shape[0]):
    print "X_training shape must match y_training shape"
print "Generate X_test and y_test"
n_input = 11
print "X_test..."
n_sample2 = np.asarray([[raw_data.ix[t-i][4] for i in range(1,n_input)] for t in np.arange (289*400,289*410)])
print "y_test..."
n_test2 =  np.asarray([raw_data.ix[t][4] for t in np.arange(289*400+1,289*410+1)])
# <codecell>
print "Training time!!!!"
nn.fit(X_training,y_training)
#
# # <codecell>
#
n_input = 11
n_sample2 = np.asarray([[raw_data.ix[t-i][4] for i in range(1,n_input)] for t in np.arange (289*400,289*410)])
n_test2 =  np.asarray([raw_data.ix[t][4] for t in np.arange(289*400+1,289*410+1)])
print nn.score(n_sample2,n_test2)
#
# # <codecell>
#
# pred = np.asarray(nn.predict(n_sample2))
# ax = pl.subplot()
# ax.set_color_cycle(['blue','red'])
# pl.plot(n_test2)
# pl.plot(pred)
# pl.show()
#
# # <codecell>
#
# pd.DataFrame(zip(pred,n_test2),columns=["Prediction","Real"])
#
# # <codecell>
#
print "Saving variable nn"
pickle.dump(nn,open('nn.pkl','wb'))

