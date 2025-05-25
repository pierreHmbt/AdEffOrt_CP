# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.autograd import Variable

# ==========================================
# Smooth approximation of quantile
# ==========================================
def Tau_e(y, epsi=.1):
    gamma = 15/16*(-1/5*(y/epsi)**5 + 2/3*(y/epsi)**3 - (y/epsi) + 8/15)
    return gamma*(y >= -epsi)*(y < epsi) + 1*(y<=-epsi)

def Tau_e_deriv(y, epsi=.1):
    gamma_deriv = 15/16*(epsi**2-y**2)**2/epsi**5
    return gamma_deriv*(y >= -epsi)*(y < epsi)

# ==========================================
# Specific to Neural Net
# ==========================================
def fNN(x, model):
    x = torch.tensor(x, dtype=torch.float32)
    return model(x).detach().numpy().ravel()

def fNN_derive(x, model):
    x = torch.tensor(x, dtype=torch.float32)
    
    # Calculate dummy gradients
    model.zero_grad()
    model(x).backward()
    grads = []
    for param in model.parameters():
        grads.append(param.grad.view(-1))
    grads = torch.cat(grads)
    return grads.detach().numpy()

def scoreNN(x, y, model):
    return np.abs(y-fNN(x, model))

def scoreNN_deriv(x, y, model):
    return -np.sign(y - fNN(x, model))*fNN_derive(x, model)

# Standard gradient descent
def QAE_NN_smooth_minibatch(X, y, model, batch_size=20, alpha = .1, stepsize = 0.01, n_iter = 1000, epsi = .1):

    optim_track = np.zeros(n_iter)
    
    weights = []
    for param in model.parameters():
        weights.append(param.view(-1).detach().numpy())

    for t in np.arange(n_iter):

        ind = np.random.choice(len(y), size=batch_size, replace=False)
        Xsub = X[ind]
        ysub = y[ind]
        
        scores = scoreNN(Xsub, ysub, model)
        k = int(np.ceil( (len(scores) + 1)*(1-alpha) ))
        quant = np.sort(scores)[k-1]
        
        s1 = 0
        s2 = 0
        for i in range(len(scores)):
            a = Tau_e_deriv(scores[i] - quant, epsi)
            s1 += a * scoreNN_deriv(Xsub[i], ysub[i], model)
            s2 += a
        grad = s1/s2

        with torch.no_grad():  
            i = 0
            sss = 0
            for param in model.parameters():
                param.data = torch.tensor(param.data - stepsize[t]*grad[sss:sss+len(weights[i])].reshape((param.size())), dtype=torch.float32)
                sss += len(weights[i])
                i += 1

        optim_track[t] = quant
    return optim_track

# Gradient descent with Adadelta
def QAE_NN_smooth_minibatch_Adadelta(X, y, model, batch_size=20, alpha = .1, n_iter = 10000, epsi = .1, rho = 0.9, epsilon = 0.0001):

    optim_track = np.zeros(n_iter)
    
    weights = []
    for param in model.parameters():
        weights.append(param.view(-1).detach().numpy())
    
    weights2 = np.concatenate(weights).ravel()
    E_grad2 = np.zeros(len(weights2))
    E_delta_para2 = np.zeros(len(weights2))
    
    for t in np.arange(n_iter):

        ind = np.random.choice(len(y), size=batch_size, replace=False)
        Xsub = X[ind]
        ysub = y[ind]
        
        scores = scoreNN(Xsub, ysub, model)
        k = int(np.ceil( (len(scores) + 1)*(1-alpha) ))
        quant = np.sort(scores)[k-1]
        
        s1 = 0
        s2 = 0
        for i in range(len(scores)):
            a = Tau_e_deriv(scores[i] - quant, epsi)
            s1 += a * scoreNN_deriv(Xsub[i], ysub[i], model)
            s2 += a
        grad = s1/s2
        E_grad2 = (rho * E_grad2) + ((1. - rho) * (grad ** 2))
        delta_param = (np.sqrt(E_delta_para2 + epsilon)) / (np.sqrt(E_grad2 + epsilon))*grad
        E_delta_para2 = (rho * E_delta_para2) + ((1. - rho) * (delta_param ** 2))
        
        with torch.no_grad():  
            i = 0
            sss = 0
            for param in model.parameters():
                param.data = torch.tensor(param.data - delta_param[sss:sss+len(weights[i])].reshape((param.size())), dtype=torch.float32)
                sss += len(weights[i])
                i += 1

        optim_track[t] = quant
    return optim_track

# ==========================================
# Specific to Linear regression
# ==========================================
def f(x, theta):
    return x.dot(theta)

def f_derive(x, theta):
    return x
    
def score(x, y, theta):
    return np.abs(y-f(x, theta).ravel())

def score_deriv(x, y, theta):
    return -np.sign(y - f(x, theta).ravel())*f_derive(x, theta)

def QAE_linear_smooth(X, y, alpha = .1, stepsize = 0.01, n_iter = 10000, epsi = .1):

    X_tilde = X.copy() #np.concatenate((np.ones((X.shape[0],1)),X), axis = 1)
    optim_track = np.zeros(n_iter)
    
    theta = np.zeros((X_tilde.shape[1],1))
    # theta = np.random.normal(size=X.shape[1]).reshape(X.shape[1], 1)
    thetamean = theta
    for t in np.arange(n_iter):

        scores = score(X_tilde, y, theta)
        k = int(np.ceil( (len(scores) + 1)*(1-alpha) ))
        quant = np.sort(scores)[k-1]
        
        s1 = 0
        s2 = 0
        for i in range(len(scores)):
            a = Tau_e_deriv(scores[i] - quant, epsi)
            s1 += a * score_deriv(X_tilde[i], y[i], theta)
            s2 += a
        grad = s1/s2

        theta = theta - stepsize[t]*grad.reshape((-1,1))
        thetamean = t/(t+1)*thetamean + theta/(t+1)
        # optim_track[t] = np.linalg.norm(grad)
        optim_track[t] = np.linalg.norm(quant)

    return np.array(theta), optim_track   