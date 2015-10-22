# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import numpy as np
import pandas as pd
import matplotlib.pyplot as pl
import sklearn
from sklearn import linear_model as lm
from sknn.mlp import Regressor, Layer
import pickle

# <codecell>

raw_data = pd.read_csv("/home/nhuan/MyWorking/tpds-2012-workload.csv");
nn = lm.RidgeCV(alphas=[0.01, 0.1,1.0,10.0])
n_row = raw_data.icol(1).count()
n_input = 10
n_range = 28900

# <codecell>

X_training = np.asarray([[raw_data.ix[t-i][4] for i in range(1,n_input)]
             for t in np.arange(n_input-1,n_range+5)])
y_training = np.asarray([raw_data.ix[t][4] for t in np.arange(n_input-1,n_range+5)])
nn.fit(X_training,y_training)

# <codecell>

n_sample2 = np.asarray([[raw_data.ix[t-i][4] for i in range(1,n_input)] for t in np.arange (30000,30400)])
n_test2 =  np.asarray([raw_data.ix[t][4] for t in np.arange(30001,30401)])
nn.score(n_sample2,n_test2)

# <codecell>

pred = np.asarray(nn.predict(n_sample2))
ax = pl.subplot()
ax.set_color_cycle(['blue','red'])
pl.plot(n_test2)
pl.plot(pred)
pl.show()

# <codecell>

nn.alpha_

# <codecell>

dpd.DataFrame(zip(pred,n_test2),columns=["Prediction","Real"],index=np.arange(30001,30401))

# <codecell>

a = raw_data[raw_data["VM ID"]==1]
a[a["Time Frame"]==1]

# <codecell>


# <codecell>


