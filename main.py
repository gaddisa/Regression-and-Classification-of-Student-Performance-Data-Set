# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 16:46:54 2019

@author: gaddi
"""
import pandas as pd
from matplotlib import pyplot
import numpy as np
import LinearRegression_1b as one_b
import RegularizedLinearRegression_1c as one_c
import RegularizedBiasedLinearRegression_1d as one_d
import BayesianLinearRegression_1e as one_e
#do the normalization using mean and standard deviation 

from preprocessing import *
#plot the comparison of all models  from 1b,1c,1d,1e
def plot_1f(yhat_b,yhat_c,yhat_d,yhat_e):
    x=np.arange(0,200)
    pyplot.figure(figsize=(12,9))
    pyplot.style.use('fivethirtyeight')
    #pyplot.ylim(-50,50)
    pyplot.plot(x, test_set_y,label='Ground Truth')
    pyplot.plot(x, yhat_b, label='('+str(rmse_b.round(decimals=2))+') Linear Regression')
    pyplot.plot(x, yhat_c,label='('+str(rmse_c.round(decimals=2))+') Linear Regression (with reg)')
    pyplot.plot(x, yhat_d, label='('+str(rmse_d.round(decimals=2))+') Linear Regression (r/b)')
    pyplot.plot(x, yhat_e, label='('+str(rmse_e.round(decimals=2))+') Bayesian Linear Regression (r/b)')

    pyplot.xlabel('Sample Index')
    pyplot.ylabel('Values')
    pyplot.title('Comparison of Linear Regression Model answer to 1f')
    pyplot.legend(loc="best")
if __name__ == '__main__':
    #partition it to 80% trainingset and 20%testset#total parameter=24
    datasets = pd.read_csv('train.csv')
    # Set seed so we get same random allocation on each run of code
    #np.random.seed(7)
    preprocessed=preprocessing(datasets)
    train_set_x,train_set_y,test_set_x,test_set_y=train_test_split_preprocessed(preprocessed)
    
    #solution1: Normal Linear Regression (1b)
    rmse_b,yhat_b=one_b.linear_regression_1b(train_set_x,train_set_y,test_set_x,test_set_y)

    #solution2: Regularized Linear Regression (1c)
    rmse_c,yhat_c=one_c.Regularized_LinearRegression_1c(train_set_x,train_set_y,test_set_x,test_set_y)
    
    #solution3: Regularized Linear Regression with bias term (1d)
    rmse_d,yhat_d=one_d.Regularized_Biased_LinearRegression_1d(train_set_x,train_set_y,test_set_x,test_set_y)
    
    #solution4: Regularized Linear Regression with bias term (1e)
    rmse_e,yhat_e=one_e.Bayesian_LinearRegression_1e(train_set_x,train_set_y,test_set_x,test_set_y)
    #print(rmse_e)
    
    #plot the comparison 1f
    plot_1f(yhat_b,yhat_c,yhat_d,yhat_e)