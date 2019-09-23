# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 16:54:36 2019

@author: gaddi
"""

import numpy as np
   
def Bayesian_LinearRegression_1e(train_set_x,training_output,test_set_x,test_set_y):
    
    number0fcolumn=len(training_output)
    
    #create ones vector for the bias term
    ones_bias=np.ones(number0fcolumn)
    
    #combine it with the training set, and now dim becomes 25
    training_data=np.column_stack((ones_bias,train_set_x))
    # Bayesian component
    # Calculate the bayesian term which is Beta is N(0,I) becouse 1/1=1
    bayes_term = np.identity(len(training_data[1,:]))
    
    #estimate the value of theta using Bayesian model
    allthethas = estimate_bayesian_model(training_output,training_data,bayes_term)
    
    weight=allthethas[1:]
    bias=allthethas[0]

    yhat=test_set_x.dot(weight)+bias
    
    rmse=np.sqrt(np.mean((yhat-test_set_y)**2))
    
    
    return rmse,yhat

#estimate the value of theta using Bayesian model
def estimate_bayesian_model(training_output,training_data,bayes):
    xT=np.matrix.transpose(training_data)
    thethas=np.dot(np.linalg.inv(np.add(np.dot(xT,training_data),bayes)),np.dot(xT,training_output))
    return thethas