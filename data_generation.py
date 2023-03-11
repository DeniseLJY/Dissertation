#!/usr/bin/env python
# coding: utf-8


import numpy as np

# Generate VAR(1) process with coefficient matrix equals to phi*Identity matrix
# and Var(error) = (1-phi^2)*Identity matrix. Thus Var[VAR(1)] = Identity matrix
def simulate_var(phi=0.1, nsimulations=1000):
    
    initialization = np.random.uniform(low=1.0, high=10.0, size=(12,))
    # AR coefficient matrix
    cov = phi*np.identity(12)
    
    # generate time series
    process = np.zeros((nsimulations,12))
    process[0,:] = initialization
    
    # generate errors
    error_mean = np.zeros(12)
    error_cov = (1-phi**2)*np.identity(12)
    errors = np.random.multivariate_normal(error_mean,error_cov,size=nsimulations)
    
    for i in range(1,nsimulations):
        process[i,:] = np.dot(cov,process[i-1,:]) + errors[i,:]
    return process


# Generate noise uncorrelated temporally but correlated across fibers
def generate_noise_1(d_1,d_2,d_3,nsimulations=1000):
    # Generate noise terms correlated cross all fibers
    noise = np.random.multivariate_normal(np.zeros(d_1),np.identity(d_1),size=(d_2*d_3*nsimulations,))
    core_noise = noise.reshape(nsimulations,d_3,d_2,d_1)
    
    # Linear transformation
    U_1 = (1/d_1)*np.ones(d_1*d_1).reshape(d_1,d_1) + np.diag((1-1/d_1)*np.ones(d_1))
    U_2 = (1/d_2)*np.ones(d_2*d_2).reshape(d_2,d_2) + np.diag((1-1/d_2)*np.ones(d_2))
    U_3 = (1/d_3)*np.ones(d_3*d_3).reshape(d_3,d_3) + np.diag((1-1/d_3)*np.ones(d_3))
    
    # We see that A × n U can be thought of as replacing each mode-n filament x by Ux
    # mode-1 product
    # we need deep copy here!!
    noise_1 = core_noise.copy()
    for j in range(d_1):
        noise_1[:,:,:,j] = core_noise[:,:,:,0] * U_1[j,0]
        for i in range(1,d_1):
            noise_1[:,:,:,j] += core_noise[:,:,:,i] * U_1[j,i]
                
    # mode-2 product
    noise_2 = noise_1.copy()
    for j in range(d_2):
        noise_2[:,:,j,:] = noise_1[:,:,0,:] * U_2[j,0]
        for i in range(1,d_2):
            noise_2[:,:,j,:] += noise_1[:,:,i,:] * U_2[j,i]
    
    # mode-3 product
    noise_3 = noise_2.copy()
    for j in range(d_3):
        noise_3[:,j,:,:] = noise_2[:,0,:,:] * U_3[j,0]
        for i in range(1,d_3):
            noise_3[:,j,:,:] += noise_2[:,i,:,:] * U_3[j,i]
    
    return(noise_3)



# setting one: core tensors are correlated temporally, noise term is uncorrelated temporally but correlated across fibers
def setting_1(d_1,d_2,d_3,phi,a_1,a_2,a_3,nsimulations=1000):
    #-------------------------------------------------
    # parameters: d_1,d_2,d_3: dimensions of tensor
    # a_1, a_2, a_3 ; loading matrix
    # phi: represent the signal strength
    #--------------------------------------------------
    
    # generate core tensor
    vectors = simulate_var(phi, nsimulations)
    factor = vectors.reshape(nsimulations,2,2,3)
    
    # loading matrices
    #a_1 = np.random.uniform(low=-2.0, high=2.0, size=(d_1,3))
    #a_2 = np.random.uniform(low=-2.0, high=2.0, size=(d_2,2))
    #a_3 = np.random.uniform(low=-2.0, high=2.0, size=(d_3,2))
    
    # mode-1 product with matrix a_1
    tensor_1 = np.zeros(d_1*2*2*nsimulations).reshape(nsimulations,2,2,d_1)
    for j in range(d_1):
        tensor_1[:,:,:,j] = factor[:,:,:,0] * a_1[j,0]
        for i in range(1,3):
            tensor_1[:,:,:,j] += factor[:,:,:,i] * a_1[j,i]
                
    # mode-2 product with matrix a_2
    tensor_2 = np.zeros(d_1*d_2*2*nsimulations).reshape(nsimulations,2,d_2,d_1)
    for j in range(d_2):
        tensor_2[:,:,j,:] = tensor_1[:,:,0,:] * a_2[j,0]
        for i in range(1,2):
            tensor_2[:,:,j,:] += tensor_1[:,:,i,:] * a_2[j,i]
    
    # mode-3 product with matrix a_3
    tensor_3 = np.zeros(d_1*d_2*d_3*nsimulations).reshape(nsimulations,d_3,d_2,d_1)
    for j in range(d_3):
        tensor_3[:,j,:,:] = tensor_2[:,0,:,:] * a_3[j,0]
        for i in range(1,2):
            tensor_3[:,j,:,:] += tensor_2[:,i,:,:] * a_3[j,i]
    
    signal_part = tensor_3 
    # add noise term
    tensor = tensor_3 + generate_noise_1(d_1,d_2,d_3,nsimulations)
    
    return [tensor,signal_part]


# Generate noise correlated temporally but uncorrelated across fibers
# We simulate noise term also from VAR(1) process
def generate_noise_2(phi,d_1,d_2,d_3,nsimulations=1000):
        
    d = d_1*d_2*d_3
    initialization = np.random.uniform(low=-1.0, high=1.0, size=(d,))
    # AR coefficient matrix
    cov = phi*np.identity(d)
    
    # generate time series
    process = np.zeros((nsimulations,d))
    process[0,:] = initialization
    
    # generate errors
    error_mean = np.zeros(d)
    error_cov = (1-phi**2)*np.identity(d)
    errors = np.random.multivariate_normal(error_mean,error_cov,size=nsimulations)
    
    for i in range(1,nsimulations):
        process[i,:] = np.dot(cov,process[i-1,:]) + errors[i,:]
    
    noise_term = process.reshape(nsimulations,d_3,d_2,d_1)
    return noise_term


# setting two: core tensors are correlated temporally with "signal strength = 0.6"
# noise term is weakly correlated temporally but uncorrelated across fibers
def setting_2(d_1,d_2,d_3,phi,a_1,a_2,a_3,nsimulations=1000):
    #-------------------------------------------------
    # phi: represent the correlation within noise series
    #--------------------------------------------------
    
    # generate core tensor
    vectors = simulate_var(0.6, nsimulations)
    factor = vectors.reshape(nsimulations,2,2,3)
    
    # generate loading matrices
    # a_1 = np.random.uniform(low=-2.0, high=2.0, size=(d_1,3))
    # a_2 = np.random.uniform(low=-2.0, high=2.0, size=(d_2,2))
    # a_3 = np.random.uniform(low=-2.0, high=2.0, size=(d_3,2))
    
    # mode-1 product with matrix a_1
    tensor_1 = np.zeros(d_1*2*2*nsimulations).reshape(nsimulations,2,2,d_1)
    for j in range(d_1):
        tensor_1[:,:,:,j] = factor[:,:,:,0] * a_1[j,0]
        for i in range(1,3):
            tensor_1[:,:,:,j] += factor[:,:,:,i] * a_1[j,i]
                
    # mode-2 product with matrix a_2
    tensor_2 = np.zeros(d_1*d_2*2*nsimulations).reshape(nsimulations,2,d_2,d_1)
    for j in range(d_2):
        tensor_2[:,:,j,:] = tensor_1[:,:,0,:] * a_2[j,0]
        for i in range(1,2):
            tensor_2[:,:,j,:] += tensor_1[:,:,i,:] * a_2[j,i]
    
    # mode-3 product with matrix a_3
    tensor_3 = np.zeros(d_1*d_2*d_3*nsimulations).reshape(nsimulations,d_3,d_2,d_1)
    for j in range(d_3):
        tensor_3[:,j,:,:] = tensor_2[:,0,:,:] * a_3[j,0]
        for i in range(1,2):
            tensor_3[:,j,:,:] += tensor_2[:,i,:,:] * a_3[j,i]
    
    signal_part = tensor_3
    # add noise term
    tensor = tensor_3 + generate_noise_2(phi,d_1,d_2,d_3,nsimulations)
    
    return [tensor,signal_part]


# Centralized
def centralization(tensor,nsimulations=1000):
    mean = np.sum(tensor,axis=0) * (1/nsimulations)
    centered_tensor = tensor - mean
    return (centered_tensor)

