#!/usr/bin/env python
# coding: utf-8

# In[1]:


from data_generation import *
from tensor_operation import *
import math
import numpy as np


# In[2]:


# Matrix-leveled estimator
# the notations follow papers'.

def mat_leveled(centered_tensor,k):
    
    # dimension recognization
    nsimulations,d_3,d_2,d_1 = centered_tensor.shape

    if k==1:
        mat_estimator = np.zeros(d_1*d_1).reshape(d_1,d_1)
        for j in range(d_1):
            for i in range(d_1):
                mat_estimator[i,j] = np.sum(centered_tensor[:,:,:,i] * centered_tensor[:,:,:,j])
            
        sample_version = mat_estimator*(1/nsimulations)
    
    if k==2:
        mat_estimator = np.zeros(d_2*d_2).reshape(d_2,d_2)
        for j in range(d_2):
            for i in range(d_2):
                mat_estimator[i,j] = np.sum(centered_tensor[:,:,i,:] * centered_tensor[:,:,j,:])
            
        sample_version = mat_estimator*(1/nsimulations)  
    
    if k==3:
        mat_estimator = np.zeros(d_3*d_3).reshape(d_3,d_3)
        for j in range(d_3):
            for i in range(d_3):
                mat_estimator[i,j] = np.sum(centered_tensor[:,i,:,:] * centered_tensor[:,j,:,:])
            
        sample_version = mat_estimator*(1/nsimulations)  
    
    return sample_version


# In[3]:


# Integrate into one function
# Matrix_leveled estimator
def matrix_estimator(centered_tensor):
    # data: input tensor time series (has been processed)
    # d_1,d_2,d_3 : dimensions of the tensor
    # nsimulations : number of simulations(observations)
    
    result = []
    for k in range(1,4):
        statistics = mat_leveled(centered_tensor,k)
        l = statistics.shape[0]
        eigen_vals, eigen_vecs = np.linalg.eig(statistics)
        ratio = list(eigen_vals[0:l-1]/eigen_vals[1:l])
        max_index = ratio.index(max(ratio))
        estimated_r = max_index + 1
        if estimated_r > math.floor(l/3):
            estimated_r = math.floor(l/3)
            
        loading_matrix = eigen_vecs[:,0:estimated_r]
        estimator = [estimated_r,loading_matrix]
        result.append(estimator)

    return result



# In[4]:


# Tensor-structured estimator
# 1.Write a function to compute tensor covariance
def tensor_cov(centered_tensor):
    
    # dimension recognization
    nsimulations,d_3,d_2,d_1 = centered_tensor.shape
    d = d_1*d_2*d_3
    
    tensor_product = np.ones(nsimulations*(d**2)).reshape(nsimulations,d_3,d_3,d_2,d_2,d_1,d_1)
    for r_6 in range(d_3):
        for r_5 in range(d_3):
            for r_4 in range(d_2):
                for r_3 in range(d_2):
                    for r_2 in range(d_1):
                        for r_1 in range(d_1):
                            tensor_product[:,r_6,r_5,r_4,r_3,r_2,r_1] = centered_tensor[:,r_5,r_3,r_1] * centered_tensor[:,r_6,r_4,r_2]
    sample_version = np.sum(tensor_product,axis=0) * (1/nsimulations)
                            
    return sample_version


# In[5]:


# 2.HOOI

# 2.1 Write a function to compute column space distance between matrix A and B
def discrepancy(A,B):
    A_space = np.dot(A,np.dot(np.linalg.inv(np.dot(A.T,A)),A.T))
    B_space = np.dot(B,np.dot(np.linalg.inv(np.dot(B.T,B)),B.T))
    space = A_space - B_space
    
    eigen_vals, eigen_vecs = np.linalg.eig(np.dot(space.T,space))
    lamb = eigen_vals[0]**0.5
    
    return lamb


# In[6]:


# 2.2 HOOI algorithm
def HOOI(covariance,d_1,d_2,d_3,r_1,r_2,r_3):
    # Initialization
    A_1 = np.zeros(d_1*r_1).reshape(d_1,r_1)
    for i in range(r_1):
        A_1[i,i] = 1.0
    A_2 = np.zeros(d_2*r_2).reshape(d_2,r_2)
    for i in range(r_2):
        A_2[i,i] = 1.0
    A_3 = np.zeros(d_3*r_3).reshape(d_3,r_3)
    for i in range(r_3):
        A_3[i,i] = 1.0
        
    # Initialize count
    m = 0
    not_converge = True
    while not_converge==True:
        B = mode_56_product(mode_34_product(covariance,A_2.T,r_2),A_3.T,r_3)
        if (m % 2) == 0:
            U,s,V = np.linalg.svd(tensor_unfold(B,2))
            A_1_hat = U[:,0:r_1]
        else:
            U,s,V = np.linalg.svd(tensor_unfold(B,1))
            A_1_hat = U[:,0:r_1]
        distance_1 = discrepancy(A_1_hat,A_1)
        A_1 = A_1_hat
        
            
        C = mode_56_product(mode_12_product(covariance,A_1.T,r_1),A_3.T,r_3)
        if (m % 2) == 0:
            U,s,V = np.linalg.svd(tensor_unfold(C,4))
            A_2_hat = U[:,0:r_2]
        else:
            U,s,V = np.linalg.svd(tensor_unfold(C,3))
            A_2_hat = U[:,0:r_2]
        distance_2 = discrepancy(A_2_hat,A_2)
        A_2 = A_2_hat
            
        D = mode_34_product(mode_12_product(covariance,A_1.T,r_1),A_2.T,r_2)
        if (m % 2) == 0:
            U,s,V = np.linalg.svd(tensor_unfold(D,6))
            A_3_hat = U[:,0:r_3]
        else:
            U,s,V = np.linalg.svd(tensor_unfold(D,5))
            A_3_hat = U[:,0:r_3]
        distance_3 = discrepancy(A_3_hat,A_3)
        A_3 = A_3_hat
        
        m += 1
        if distance_1 <=0.00001 and distance_2 <=0.00001 and distance_3 <=0.00001:
            not_converge = False
            
    result = [m,A_1,A_2,A_3]
    return result   



