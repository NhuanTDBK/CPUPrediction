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

# <codecell>
nn = Regressor(
    layers=[
        Layer("Rectifier", units=10),
        Layer("Linear")],
    learning_rate=0.02,
    n_iter=30,
    debug=True,
    verbose=1)
print "Reading file......"
raw_data = pd.read_csv("tpds-2012-workload.csv");
n_row = raw_data.shape[0]
n_input = 11
data = raw_data["Utilization"]
print "Generate X_traing, y_traing"
print "X_training loading..."
X_training = np.asarray([[data[i] for i in range(1,n_input)]
             for t in np.arange(n_input-1,n_row)])
print "y_training loading..."
y_training = data[n_input-1:]

# <codecell>
print "Training time!!!!"
nn.fit(X_training,y_training)

#
# # <codecell>
#
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

