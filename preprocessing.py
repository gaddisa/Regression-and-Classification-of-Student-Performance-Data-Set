# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 16:01:21 2019

@author: gaddisa olani

This modolu is intended to to perform preprocessing task: normalization, onehot encoding
"""

import pandas as pd
import numpy as np
def normilizeAttributes(df_num):
    result = df_num.copy()
    for feature_name in result.columns:
        mean_value = result[feature_name].mean()
        standard_deviation = result[feature_name].std()
        result[feature_name] = (result[feature_name] - mean_value) / (standard_deviation)
    return result
def onehot_encoding(df_categorical):
    result=df_categorical.copy()
    for feature_name in result.columns:
        one_hot = pd.get_dummies(result[feature_name])
        result = result.drop(feature_name,axis = 1)
        result = result.join(one_hot,lsuffix='_caller', rsuffix='_other')
    return result

def preprocessing(datasets):
    #use only attributes specified in the question, and screen out the others
    filtered_datasets=datasets[['school','sex','age','famsize','studytime','failures','activities'\
                                ,'higher','internet','romantic','famrel','freetime','goout'\
                                ,'Dalc','Walc','health','absences','G3']]
    
    #select numeric attributes and apply normalization
    df_all_numeric = filtered_datasets.select_dtypes(include=[np.number])
    
    #normalization is not needed for G3
    df_num,df_G3=df_all_numeric[df_all_numeric.columns.difference(['G3'])],df_all_numeric.loc[:,'G3']
    afterNormalization=normilizeAttributes(df_num)
    
    #select categorical attribute and convert it to one hot encoding
    df_categorical=filtered_datasets.select_dtypes(include=['object'])
    afteronehotencoded=onehot_encoding(df_categorical)
    
    #combine the normalized and onehot encoded result
    preprocessed_data=pd.concat([afterNormalization,afteronehotencoded,df_G3],axis=1)
    return preprocessed_data

def train_test_split_preprocessed(preprocessed):
    
    #to avoid overfitting, randomize the training samples used for model construction
    X=preprocessed.sample(frac=1)
   # X=preprocessed
    train_set_size = int(len(X) * 0.80)
    train_set, test_set = X[0:train_set_size], X[train_set_size:len(X)]
    
    #separate predictor attributes and target attributes (y=G3)
    train_set_x,train_set_y=train_set[train_set.columns.difference(['G3'])],train_set.loc[:,'G3']
    test_set_x,test_set_y=test_set[test_set.columns.difference(['G3'])],test_set.loc[:,'G3']

    return train_set_x.values,train_set_y.values,test_set_x.values,test_set_y.values

