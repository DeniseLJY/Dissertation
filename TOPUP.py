#!/usr/bin/env python
# coding: utf-8



from data_generation import *
import math
import numpy as np



# TOPUP Method
# the notations follow papers'.
# h_0 = 1 because we generate VAR(1) process

# -------------------------
# some preparation for calcuating TOPUP_K:
# 1. tensor unfolding: unflod (nsimulations,d_3,d_2,d_1)tensor into (nsimulations,d_1,d_2*d_3)
# (nsimulations,d_2,d_1*d_3) and (nsimulations,d_3,d_1*d_2)
# 2. tensor product
# -------------------------



# 1. tensor unfolding
def tensor_unfolding(tensor, k):
    #-------
    # k:mode-k unfolding
    #-------
    nsimulations,d_3,d_2,d_1 = tensor.shape
    if k == 1:
        t_1 = np.swapaxes(tensor, 3, 1)
        mat = np.reshape(t_1, newshape=(nsimulations,d_1,d_2*d_3), order='F')
    if k == 2:
        t_1 = np.swapaxes(tensor, 2, 1)
        t_2 = np.swapaxes(t_1, 3, 2)
        mat = np.reshape(t_2, newshape=(nsimulations,d_2,d_1*d_3), order='F')
    if k == 3:
        t_1 = np.swapaxes(tensor, 3, 2)
        mat = np.reshape(t_1, newshape=(nsimulations,d_3,d_1*d_2), order='F')
    
    return (mat)



# 2. tensor product
def tensor_product_1(mat):
    
    # determine dimensions
    nsimulations,d_k,d_ = mat.shape
    mat_1 = mat[:-1,:,:]
    mat_2 = mat[1:,:,:]
    
    tensor_product = np.ones((nsimulations-1)*(d_k*d_)**2).reshape((nsimulations-1),d_,d_k,d_,d_k)
    for r_4 in range(d_):
        for r_3 in range(d_k):
            for r_2 in range(d_):
                for r_1 in range(d_k):
                    tensor_product[:,r_4,r_3,r_2,r_1]= mat_1[:,r_1,r_2]*mat_2[:,r_3,r_4]
    
    return tensor_product



# calculate TOPUP_K
def topup_k(data,k):
    nsimulations,d_3,d_2,d_1 = data.shape
    mat = tensor_unfolding(data, k)
    tensor_product = tensor_product_1(mat)
    topup = np.sum(tensor_product,axis=0) * (1/(nsimulations-1))
    return (topup)


# ----------------------------
# calculate W_k and operate eigenvalue decomposition
# ----------------------------
def W_k(topup_k):
    # dimension recognization
    d_,d_k = topup_k.shape[-2:]
    # mode-1 unfolding
    mat_1  = (topup_k.reshape(d_*d_*d_k,d_k)).T
    w_k = np.dot(mat_1,mat_1.T)
    
    return w_k

def eigen_decomposition(w_k):
    # the output is the loading matrix A_k
    l = w_k.shape[0]
    eigen_vals, eigen_vecs = np.linalg.eig(w_k)
    ratio = list(eigen_vals[0:l-1]/eigen_vals[1:l])
    max_index = ratio.index(max(ratio))
    estimated_r = max_index + 1
    if estimated_r > math.floor(l/3):
        estimated_r = math.floor(l/3)
    
    loading_matrix = eigen_vecs[:,0:estimated_r]
    
    return [estimated_r,loading_matrix]


# Integrate into one function
def TOPUP_Method(centered_tensor):
    # centered_tensor: input tensor time series (has been processed)
    
    result = []
    for k in range(1,4):
        top_k = topup_k(centered_tensor,k)
        w_k = W_k(top_k)
        result.append(eigen_decomposition(w_k))
        
    return result







