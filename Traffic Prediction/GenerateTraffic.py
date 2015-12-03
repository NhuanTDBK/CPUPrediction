
# coding: utf-8

# In[46]:

import pandas as pd
import numpy as np
from pandas import HDFStore


# In[53]:

raw_data_name = "ita_public_tools/output/data.csv"
raw_data = pd.read_csv(raw_data_name)
store = HDFStore("storeTraffic.h5")


# In[62]:

data = raw_data.groupby('Timestamp').count()["Timestamp"]
store["conn"] = data


# In[49]:

# n_input = 10
# n_row = data.shape[0]
# print "Generate X_training, y_training"
# print "X_training loading..."
# X_training = np.asarray([[data[t-i-1] for i in range(0,n_input)]
#              for t in np.arange(n_input,n_row)])


# In[ ]:



