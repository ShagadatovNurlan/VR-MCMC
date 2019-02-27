#Linear regression for G_pml + explicit formula for polynomial approximation

import numpy as np
from scipy.misc import comb
from scipy.special import hermitenorm
from tqdm import tqdm
from joblib import Parallel, delayed
from itertools import product
from sklearn.preprocessing import PolynomialFeatures
from matplotlib import pyplot as plt
import math
import plotly.plotly as py
import plotly.graph_objs as go
import plotly

def H(k, x):
    if k==0:
        return 1.0
    if k ==1:
        return x
    if k==2:
        return (x**2 - 1)/np.sqrt(2)
    h = hermitenorm(k)(x) /  np.sqrt(math.factorial(k))
    return h

def Hermite_val(k_vec,x_vec):
    P = 1.0
    d = x_vec.shape[0]
    for i in range(d):
        P = P * H(k_vec[i],x_vec[i])
    return P

def generate_X_poly(train_traj, r, max_deg):
    N_train = train_traj.shape[0]
    N = train_traj.shape[1]
    d = train_traj.shape[2]
    X = np.empty((N_train * (N-r) ,d+1 + int(d*(d+1)/2)))
    all_points = train_traj[:, :N-r].reshape(-1,d)
    poly = PolynomialFeatures(max_deg)
    X = poly.fit_transform(all_points)
    return X, poly.powers_

def generate_y_mean(train_traj, r, f_target = "sum"):
    N_train = train_traj.shape[0]
    N = train_traj.shape[1]
    y = np.zeros(N_train * (N-r))
    if f_target == "sum":
        y = train_traj[:, r:].sum(axis = 2).reshape(-1)
    elif f_target == "sum_squared":
        y = np.square(train_traj[:, r:]).sum(axis = 2).reshape(-1)
    elif f_target == "sum_4th":
        y = (train_traj[:, r:]**4).sum(axis = 2).reshape(-1)
    elif f_target == "exp_sum":
        y = np.exp(train_traj[:, r:]).sum(axis =2).reshape(-1)
    else:
        raise Exception('unrecognized target function')
    return y


def G_pml_fit_mean(train_traj, f_target="sum", max_deg = 1):
    N_train = train_traj.shape[0]
    N = train_traj.shape[1]
    d = train_traj.shape[2]
    Betas = np.zeros((N, d+ 1 + (max_deg-1) * int(d*(d+1)/2)))
    for r in tqdm(range(N)):
        # Linear Regression
        if 0 < max_deg < 3:
            X, degrees = generate_X_poly(train_traj, r, max_deg)
        else:
            raise Exception('max_deg should be 1 or 2')
        y = generate_y_mean(train_traj, r, f_target)

        beta = np.linalg.inv(X.T @ X) @ X.T @ y
        Betas[r] = beta
    return Betas, degrees

def G_pml_predict(x, pml, Betas, max_deg = 1):
    poly = PolynomialFeatures(max_deg)
    x_pol = poly.fit_transform(x.reshape(1,-1))
    beta = Betas[pml]
    return (x_pol @ beta)

def a_plk(traj, traj_grad, p, l, k_vec, step, degrees, Betas):
    dim = traj.shape[1]
    S = 0
    x_hat = traj[l-1] - step*traj_grad[l-1]
    Small_s = np.zeros(dim)
    for ind,deg in enumerate(degrees):
        Small_s[:] = 0
        for d, i in enumerate(deg):
            for t in range (i+1):
                for s in range (int(t/2 +1)):
                    if (k_vec[d] == t - 2*s):
                        Small_s[d] = Small_s[d] + comb(N=i, k = t, exact = True) * x_hat[0]**(i-t) * math.factorial(t)*1/math.factorial(s)*1 / np.sqrt(math.factorial(t-2*s)) *np.sqrt(2*step)**t /2**s
                    else:
                        pass
        S = S + Betas[p-l,ind] * Small_s.prod()
    return S

def M_bias(k_vec, traj, traj_grad, traj_noise, step, degrees, Betas, n_tilde):
    N = traj.shape[0]
    S = 0
    for p in range(N):
        for l in range (p+1):
            if (p-l<n_tilde):   #TRUNCATED
                s = a_plk(traj, traj_grad, p, l, k_vec, step, degrees, Betas)* Hermite_val(k_vec,traj_noise[l])
                S = S + s
    return S/N

def estimator_bias(k_vec, test_traj, test_traj_grad, test_traj_noise, step, degrees, Betas, n_tilde, n_jobs = -1):
    N_test = test_traj.shape[0]
    M_results = Parallel(n_jobs = n_jobs)(delayed(M_bias)(k_vec, test_traj[i], test_traj_grad[i], test_traj_noise[i], step, degrees, Betas,n_tilde)for i in range(N_test))
    return np.array(M_results).reshape(-1)
