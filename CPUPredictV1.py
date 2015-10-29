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

raw_data = pd.read_csv("/home/nhuan/MyWorking/tpds-2012-workload.csv");
nn = Regressor(
    layers=[
        Layer("Rectifier", units=10),
        Layer("Linear")],
    learning_rate=0.02,
    n_iter=30,
    debug=True)
n_row = raw_data.icol(1).count()
n_input = 11
n_range = 57

# <codecell>

X_training = np.asarray([[raw_data.ix[t-i][4] for i in range(1,n_input)]
             for t in np.arange(n_input-1,n_range+5)])
y_training = np.asarray([raw_data.ix[t][4] for t in np.arange(n_input-1,n_range+5)])
nn.fit(X_training,y_training)
#
# # <codecell>
#
# n_sample2 = np.asarray([[raw_data.ix[t-i][4] for i in range(1,n_input)] for t in np.arange (289*400,289*410)])
# n_test2 =  np.asarray([raw_data.ix[t][4] for t in np.arange(289*400+1,289*410+1)])
# nn.score(n_sample2,n_test2)
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

