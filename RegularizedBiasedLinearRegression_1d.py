# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 16:54:36 2019

@author: gaddi
"""

import numpy as np

def find_optimumweight_regularization_1d(train_set,train_output):
    y=train_output 
    y = train_output.reshape((train_output.shape[0], 1))  

    number0fcolumn=len(y)
    
    #define vector of ones for thetha zero (w0)
    ones_w0=np.ones(number0fcolumn)
    #combine it with the training set
    x=np.column_stack((ones_w0,train_set))
    transpose_x = np.matrix.transpose(x)
    # Identity matrix (number of parameters is the dimension)
    I = np.identity(len(x[1,:]))
    # We don't add penalty to intercept
    I[0,0] = 0
    lmda=0.5
    calculated_weight = np.dot(np.linalg.inv(np.add(np.dot(transpose_x,x),lmda*I)),np.dot(transpose_x,y)) 
    #now the weight vector has bias which is weight[0], and 24 other weights for each feature we have in 
    bias=calculated_weight[0] #bias term
    weight=calculated_weight[1:] #from 1 to 24 are weights
    wT=weight.T
    regularizer=lmda*(wT.dot(weight))
    
    yhat=train_set.dot(weight)+bias
    
    #weight decay formula
    rmse=np.sqrt(np.mean((yhat-train_output)**2))+regularizer
  
    return rmse,yhat,bias, weight

def get_weight_bias_1d(train_set_x, train_set_y,num_iters=100,batch_size=20, verbose=False):
    X = train_set_x
    num_train, dim = X.shape
    y=train_set_y
    #initial let say loss=1000 to keep the minimum one
    loss_history =1000.0
    tempweight=0.0
    tempbias=0.0
    
    for it in range(num_iters):
        indices = np.random.choice(num_train,batch_size,replace=False)
        X_batch = X[indices]
        y_batch = y[indices]
        #find the weight
        loss,yhat,bias,weight=find_optimumweight_regularization_1d(X_batch,y_batch)
        if loss_history>loss:
            loss_history=loss
            tempweight=weight
            tempbias=bias
    return tempweight,tempbias

def Regularized_Biased_LinearRegression_1d(train_set_x,train_set_y,test_set_x,test_set_y):
    weight,bias=get_weight_bias_1d(train_set_x, train_set_y,num_iters=100,batch_size=50, verbose=False)
    yhat=test_set_x.dot(weight)+bias
    rmse=np.sqrt(np.mean((yhat-test_set_y)**2))
    
    return rmse,yhat