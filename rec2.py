import pandas as pd
import numpy as np
import time
import turicreate as tc
from sklearn.model_selection import train_test_split

names = ['user_id','item_id','rating','timestamp']
df = pd.read_csv('ml-100k/u.data',sep='\t',names=names)
n_users = df.user_id.unique().shape[0]
n_items = df.item_id.unique().shape[0]

ratings = np.zeros((n_users,n_items))

for row in df.itertuples():
    ratings[row[1] - 1, row[2] - 1] = row[3]


sparsity = float(len(ratings.nonzero()[0]))
sparsity /= (ratings.shape[0] * ratings.shape[1])
sparsity *= 100

def train_test_split(ratings):
    train = ratings.copy()
    test = np.zeros(ratings.shape)
    for user in range(ratings.shape[0]):
        # indices of non zero get item id
        test_ratings = np.random.choice(ratings[user,:].nonzero()[0],size=10,replace=False)
        train[user,test_ratings] = 0
        test[user,test_ratings] = ratings[user,test_ratings]
        return train,test


train,test = train_test_split(ratings)