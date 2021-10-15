import pandas as pd
import os
import sys
import pickle
from sklearn import preprocessing

dataset = pd.read_pickle(open(os.path.join(sys.path[0], "sampled_health_dataset.csv"), "rb"))
col =dataset.columns
dataset = preprocessing.scale(dataset)
dataset = pd.DataFrame(dataset,columns=col).round(decimals=2)
dataset['age'][dataset['age']> 0] = 1
dataset['age'][dataset['age'] < 0] = 0
dataset['labels'][dataset['labels'] > 0] = 1
dataset['labels'][dataset['labels'] < 0] = 0
print(dataset)

dataset.to_pickle('scaled_health_dataset.csv')


