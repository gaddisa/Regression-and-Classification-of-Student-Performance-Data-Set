# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 18:59:48 2019

@author: gaddi
"""
import numpy as np
from numpy.linalg import inv

#implementation of regularized LR, withour regularization, question 1b
def linear_regression_1b(train_set_x,train_set_y,test_set_x,test_set_y):
    #use a pseduo inverse matrix to find the weight (coefficient of regression)
    x = train_set_x
    y = train_set_y.reshape((train_set_y.shape[0], 1))  
    weight = inv(x.T.dot(x)).dot(x.T).dot(y)
    
    yhat=test_set_x.dot(weight)
    rmse=np.sqrt(np.mean((yhat-test_set_y)**2))

    return rmse,yhat 

