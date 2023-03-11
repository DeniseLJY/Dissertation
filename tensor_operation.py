#!/usr/bin/env python
# coding: utf-8


# two functions that apply on tensor
# mode_k product with matrix
# mode_k unfolding

import numpy as np

def mode_12_product(tensor,matrix,r_1):
    # compute mode_1 and mode_2 product
    # count dimensions
    d_6,d_5,d_4,d_3,d_2,d_1 = tensor.shape
    d = d_6*d_5*d_4*d_3*d_2*r_1
    n = d_6*d_5*d_4*d_3*r_1*r_1
    
    product_1 = np.zeros(d).reshape(d_6,d_5,d_4,d_3,d_2,r_1)
    for i in range(r_1):
        for j in range(d_1):
            product_1[:,:,:,:,:,i] += tensor[:,:,:,:,:,j]*matrix[i,j]
    
    product_2 = np.zeros(n).reshape(d_6,d_5,d_4,d_3,r_1,r_1)
    for i in range(r_1):
        for j in range(d_2):
            product_2[:,:,:,:,i,:] += product_1[:,:,:,:,j,:]*matrix[i,j]
    
    return product_2



def mode_34_product(tensor,matrix,r_2):
    # compute mode_3 and mode_4 product
    # count dimensions
    d_6,d_5,d_4,d_3,d_2,d_1 = tensor.shape
    d = d_6*d_5*d_4*r_2*d_2*d_1
    n = d_6*d_5*r_2*r_2*d_2*d_1
    
    product_3 = np.zeros(d).reshape(d_6,d_5,d_4,r_2,d_2,d_1)
    for i in range(r_2):
        for j in range(d_3):
            product_3[:,:,:,i,:,:] += tensor[:,:,:,j,:,:]*matrix[i,j]
    
    product_4 = np.zeros(n).reshape(d_6,d_5,r_2,r_2,d_2,d_1)
    for i in range(r_2):
        for j in range(d_4):
            product_4[:,:,i,:,:,:] += product_3[:,:,j,:,:,:]*matrix[i,j]
    
    return product_4



def mode_56_product(tensor,matrix,r_3):
    # compute mode_5 and mode_6 product
    # count dimensions
    d_6,d_5,d_4,d_3,d_2,d_1 = tensor.shape
    d = d_6*r_3*d_4*d_3*d_2*d_1
    n = r_3*r_3*d_4*d_3*d_2*d_1
    
    product_5 = np.zeros(d).reshape(d_6,r_3,d_4,d_3,d_2,d_1)
    for i in range(r_3):
        for j in range(d_5):
            product_5[:,i,:,:,:,:] += tensor[:,j,:,:,:,:]*matrix[i,j]
    
    product_6 = np.zeros(n).reshape(r_3,r_3,d_4,d_3,d_2,d_1)
    for i in range(r_3):
        for j in range(d_6):
            product_6[i,:,:,:,:,:] += product_5[j,:,:,:,:,:]*matrix[i,j]
    
    return product_6

# write function to unfold tensor
def tensor_unfold(tensor,k):
    # k: mode
    # count dimensions
    d_6,d_5,d_4,d_3,d_2,d_1 = tensor.shape
    
    if k == 1:
        mat_k = tensor.reshape(d_6*d_5*d_4*d_3*d_2,d_1)
        mat = mat_k.T
    if k == 2:
        mat = tensor[:,:,:,:,0,:].flatten()
        for i in range(1,d_2):
            vec = tensor[:,:,:,:,i,:].flatten()
            mat = np.vstack((mat,vec))
    if k == 3:
        mat = tensor[:,:,:,0,:,:].flatten()
        for i in range(1,d_3):
            vec = tensor[:,:,:,i,:,:].flatten()
            mat = np.vstack((mat,vec))
    if k == 4:
        mat = tensor[:,:,0,:,:,:].flatten()
        for i in range(1,d_4):
            vec = tensor[:,:,i,:,:,:].flatten()
            mat = np.vstack((mat,vec))
    if k == 5:
        mat = tensor[:,0,:,:,:,:].flatten()
        for i in range(1,d_5):
            vec = tensor[:,i,:,:,:,:].flatten()
            mat = np.vstack((mat,vec))
    if k == 6:
        mat = tensor[0,:,:,:,:,:].flatten()
        for i in range(1,d_6):
            vec = tensor[i,:,:,:,:,:].flatten()
            mat = np.vstack((mat,vec))
            
    return mat
