import numpy as np 
import pandas as pd 
import os

df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')
#print(train.head())
#print(train.isnull().sum().any())
for col in df_train.columns:
        if df_train[col].dtype == 'object' and len(df_train[col].unique()) >= 10:
            df_train.drop(columns=[col], inplace=True)
df_train.fillna(0, inplace=True)
#test = df_test.fillna(0, inplace=True)
train = pd.get_dummies(df_train)
#test = pd.get_dummies(test)
train.to_csv('trainpro.csv',index=False)
for col in df_test.columns:
        if df_test[col].dtype == 'object' and len(df_test[col].unique()) >= 10:
            df_test.drop(columns=[col], inplace=True)

df_test.fillna(0, inplace=True)
test = pd.get_dummies(df_test)
test.to_csv('testpro.csv', index=False)
print('ok')
