# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 17:11:31 2019

@author: gaddi
"""


import numpy as np

def find_optimumweight_regularization_1c(train_set_x,train_set_y):
    y=train_set_y 
    y = train_set_y.reshape((train_set_y.shape[0], 1))  

    number0fcolumn=len(y)
    
    #define vector of ones for thetha zero (w0)
    ones_w0=np.ones(number0fcolumn)
    #combine it with the training set
    x=np.column_stack((ones_w0,train_set_x))
    transpose_x = np.matrix.transpose(x)
    # Identity matrix (number of parameters is the dimension)
    I = np.identity(len(x[1,:]))
    # We don't add penalty to intercept
    I[0,0] = 0
    lmda=0.5
    # transpose the training example to make it suitable for multiplication 
    calculated_weight = np.dot(np.linalg.inv(np.add(np.dot(transpose_x,x),lmda*I)),np.dot(transpose_x,y)) 
    
    #avoid bias which is calcualed_weight[0]
    weight=calculated_weight[1:]
    wT=weight.T
    regularizer=lmda*(wT.dot(weight))
    
    yhat=train_set_x.dot(weight)
    
    #weight decay formula
    rmse=np.sqrt(np.mean((yhat-train_set_y)**2))+regularizer
  
    return rmse,yhat,weight

def get_weight_1c(train_set_x, train_set_y,num_iters=2,batch_size=20, verbose=False):
    X = train_set_x
    num_train, dim = X.shape
    y=train_set_y
    #initial let say loss=1000 to keep the minimum one
    loss_history =1000.0
    weight=0.0    
    for it in range(num_iters):
        indices = np.random.choice(num_train,batch_size,replace=False)
        X_batch = X[indices]
        y_batch = y[indices]
        #find the weight
        loss,yhat,temp_weight=find_optimumweight_regularization_1c(X_batch,y_batch)
        if loss_history>loss:
            loss_history=loss
            weight=temp_weight
    return weight

def Regularized_Biased_LinearRegression_1c(train_set_x,train_set_y,test_set_x,test_set_y):
    weight=get_weight_1c(train_set_x, train_set_y,num_iters=100,batch_size=100, verbose=False)
    yhat=test_set_x.dot(weight)
    rmse=np.sqrt(np.mean((yhat-test_set_y)**2))
    
    return rmse,yhat