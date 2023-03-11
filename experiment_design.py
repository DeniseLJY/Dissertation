#!/usr/bin/env python
# coding: utf-8

# In[1]:


from my_method import *
from TOPUP import *
from data_generation import *
from tensor_operation import *
import math
import numpy as np

import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


## Metric one
def metric_one(result, a_1, a_2, a_3, rep=10):
    # --------------------
    # a_1, a_2, a_3 are real loading matrics
    # --------------------
    
    A1 = np.zeros(rep*3).reshape(rep, 3)
    A2 = np.zeros(rep*3).reshape(rep, 3)
    A3 = np.zeros(rep*3).reshape(rep, 3)
    
    for i in range(rep):
        A1[i,:] = np.array([discrepancy(result[i][1][0][1],a_1),discrepancy(result[i][2][0][1],a_1),
                            discrepancy(result[i][3][1],a_1)])
        A2[i,:] = np.array([discrepancy(result[i][1][1][1],a_2),discrepancy(result[i][2][1][1],a_2),
                            discrepancy(result[i][3][2],a_2)])
        A3[i,:] = np.array([discrepancy(result[i][1][2][1],a_3),discrepancy(result[i][2][2][1],a_3),
                            discrepancy(result[i][3][3],a_3)])
    
    # Draw box plot
    A_1 = {
        'TOPUP': list(A1[:,0]),
        'Matrix_Estimator': list(A1[:,1]),
        'Tensor_Estimator': list(A1[:,2])
    }
    df_1 = pd.DataFrame(A_1)
    A_2 = {
        'TOPUP': list(A2[:,0]),
        'Matrix_Estimator': list(A2[:,1]),
        'Tensor_Estimator': list(A2[:,2])
    }
    df_2 = pd.DataFrame(A_2)
    A_3 = {
        'TOPUP': list(A3[:,0]),
        'Matrix_Estimator': list(A3[:,1]),
        'Tensor_Estimator': list(A3[:,2])
    }
    df_3 = pd.DataFrame(A_3)
    
    df_1.plot.box(title="Estimated A_1")
    df_2.plot.box(title="Estimated A_2")
    df_3.plot.box(title="Estimated A_3")
    plt.title('Estimation:d_1=10,d_2=10,d_3=10,observation=1000')
    plt.show()


# In[3]:


## Metric Two
# Firstly write a function to compute the estimated signal part and the following error
def signal_part(data, a_1,a_2,a_3, n_sample=50, interval=20):
    # ----------------------------
    # n_sample: the number of "point" sampled from real tensor time series
    # interval = [nsimulations/n_sample]
    # data: [real tensor time series, signal part]
    # a_1, a_2, a_3; estimated loading matrices
    # ____________________________
    
    # dimension recognition
    nsimulations,d_3,d_2,d_1 = data[0].shape
    r_1 = a_1.shape[1]
    r_2 = a_2.shape[1]
    r_3 = a_3.shape[1]
    
    # uniform random sampling
    sample_index = range(1,nsimulations,interval)
    sampled_data = data[0][sample_index,:,:,:]
    sampled_signal = data[1][sample_index,:,:,:]
    
    # compute factors
    product_1 = np.zeros(r_1*d_2*d_3*n_sample).reshape(n_sample,d_3,d_2,r_1)
    for i in range(r_1):
        for j in range(d_1):
            product_1[:,:,:,i] += sampled_data[:,:,:,j] * a_1[j,i]
            
    product_2 = np.zeros(r_1*r_2*d_3*n_sample).reshape(n_sample,d_3,r_2,r_1)
    for i in range(r_2):
        for j in range(d_2):
            product_2[:,:,i,:] += product_1[:,:,j,:] * a_2[j,i]
    
    factor = np.zeros(r_1*r_2*r_3*n_sample).reshape(n_sample,r_3,r_2,r_1)        
    for i in range(r_3):
        for j in range(d_3):
            factor[:,i,:,:] += product_2[:,j,:,:] * a_3[j,i]
            
    # compute estimated signal part
    signal_1 = np.zeros(d_1*r_2*r_3*n_sample).reshape(n_sample,r_3,r_2,d_1)
    for i in range(d_1):
        for j in range(r_1):
            signal_1[:,:,:,i] += factor[:,:,:,j] * a_1[i,j]
    
    signal_2 = np.zeros(d_1*d_2*r_3*n_sample).reshape(n_sample,r_3,d_2,d_1)
    for i in range(d_2):
        for j in range(r_2):
            signal_2[:,:,i,:] += signal_1[:,:,j,:] * a_2[i,j]
    
    signal = np.zeros(d_1*d_2*d_3*n_sample).reshape(n_sample,d_3,d_2,d_1)
    for i in range(d_3):
        for j in range(r_3):
            signal[:,i,:,:] += signal_2[:,j,:,:] * a_3[i,j]
    
    error = sampled_signal - signal
    
    return error

# Then write a function to compute Frobenius norm of the error
def Frobenius(error, n_sample=50):
    f_norm = np.ones(50)
    f_norm_square = error * error
    for i in range(n_sample):
        f_norm[i] = np.sqrt(np.sum(f_norm_square[i,:,:,:]))
        
    return f_norm


# In[4]:


# Integrated into one function
def metric_two(result, n_samples=50, intervals=20, rep=5):
    
    f = np.zeros(rep*n_samples*3).reshape(rep, n_samples, 3)    
    for i in range(rep):
        f[i,:,0] = Frobenius(signal_part(result[i][0], result[i][1][0][1],result[i][1][1][1],result[i][1][2][1], 
                                         n_sample=n_samples, interval=intervals), n_sample=n_samples)
        f[i,:,1] = Frobenius(signal_part(result[i][0], result[i][1][0][1],result[i][1][1][1],result[i][1][2][1], 
                                         n_sample=n_samples, interval=intervals), n_sample=n_samples)
        f[i,:,2] = Frobenius(signal_part(result[i][0], result[i][3][1],result[i][3][2],result[i][3][3], 
                                         n_sample=n_samples, interval=intervals), n_sample=n_samples)
    
    F = f.reshape(rep*n_samples, 3)
    # Draw box plot
    F_norm = {
        'TOPUP': list(F[:,0]),
        'Matrix_Estimator': list(F[:,1]),
        'Tensor_Estimator': list(F[:,2])
    }
    df = pd.DataFrame(F_norm)
    df.plot.box(title="Accuracy of estimating the signal part - Frobenius norm")
    plt.title('Estimation:d_1=10,d_2=10,d_3=10,observation=1000')
    plt.show()


# In[5]:


def test_1(phi,a_1,a_2,a_3,d_1=10,d_2=10,d_3=10,nsim=1000,rep=5):
    results = []
    
    for i in range(rep):
        data = setting_1(d_1,d_2,d_3,phi,a_1,a_2,a_3, nsimulations=nsim)
        centered_tensor = centralization(data[0],nsimulations=nsim)
        method_1 = TOPUP_Method(centered_tensor)
        method_2 = matrix_estimator(centered_tensor)
        # method 3
        tensor_covariance = tensor_cov(centered_tensor)
        # r_1,r_2,r_3
        r1 = method_2[0][0]
        r2 = method_2[1][0]
        r3 = method_2[2][0]
        method_3 = HOOI(tensor_covariance,d_1,d_2,d_3,r1,r2,r3)
        
        results.append([data, method_1, method_2, method_3])
    
    return results


# In[6]:


def test_2(phi,a_1,a_2,a_3,d_1=10,d_2=10,d_3=10,nsim=1000,rep=5):
    # ------------------------
    # Note that phi here represents correlation parameter of noise process.
    # ------------------------
    results = []
    
    for i in range(rep):
        data = setting_2(d_1,d_2,d_3,phi,a_1,a_2,a_3,nsimulations=nsim)
        centered_tensor = centralization(data[0],nsimulations=nsim)
        method_1 = TOPUP_Method(centered_tensor)
        method_2 = matrix_estimator(centered_tensor)
        # method 3
        tensor_covariance = tensor_cov(centered_tensor)
        # r_1,r_2,r_3
        r1 = method_2[0][0]
        r2 = method_2[1][0]
        r3 = method_2[2][0]
        method_3 = HOOI(tensor_covariance,d_1,d_2,d_3,r1,r2,r3)
        
        results.append([data, method_1, method_2, method_3])
    
    return results

