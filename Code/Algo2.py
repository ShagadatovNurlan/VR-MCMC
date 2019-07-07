# Algorithm 1, polynomial regression for Q_l + explicit formula + truncation

import numpy as np
from scipy.misc import comb
from scipy.special import hermitenorm
from tqdm import tqdm
from joblib import Parallel, delayed
from itertools import product
from sklearn.preprocessing import PolynomialFeatures
import math

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

def generate_X_poly(train_traj, l, max_deg):
    N_train = train_traj.shape[0]
    N = train_traj.shape[1]
    d = train_traj.shape[2]
    poly = PolynomialFeatures(max_deg)
    # all_points = train_traj[:, :l+1].reshape(-1,d)
    # X = poly.fit_transform(all_points)
    X = poly.fit_transform(train_traj[:, l])
    return X, poly.powers_

def generate_y_sum(train_traj, l, f_target, n_tilde):
    N_train = train_traj.shape[0]
    N = train_traj.shape[1]
    d = train_traj.shape[2]
    y = np.zeros(N_train)
    # y = np.zeros(N_train*(l+1))
    # for s in range(N_train):
    #     for i in range(l+1):
    #         if f_target == "sum":
    #             y[s*l + i] = train_traj[s, i:i + N - l].sum()/N
    #             # y[s*l + i] = train_traj[s, i:i + n_tilde].sum()/N
    #         elif f_target == "sum_squared":
    #             y[s*l + i] = np.square(train_traj[s, i:i + N - l]).sum()/N
    #             # y[s*l + i] = np.square(train_traj[s, i:i + n_tilde]).sum()/N
    #         elif f_target == "sum_4th":
    #             y[s*l + i] = (train_traj[s, i:i + N - l]**4).sum()/N
    #         elif f_target == "exp_sum":
    #             y[s*l + i] = np.exp(train_traj[s, i:i + N - l].sum(axis =1)).sum()/N
    #         else:
    #             raise Exception('unrecognized target function')
    # return y
    for s in range(N_train):
        if f_target == "sum":
            y[s] = train_traj[s,l:].sum()/N
            # y[s] = train_traj[s,l:l+n_tilde].sum()/N
        elif f_target == "sum_squared":
            y[s] = np.square(train_traj[s, l:]).sum()/N
        elif f_target == "sum_4th":
            y[s] = (train_traj[s,l:]**4).sum()/N
        elif f_target == "exp_sum":
            y[s] = np.exp(train_traj[s, l:].sum(axis =1)).sum()/N
        else:
            raise Exception('unrecognized target function')
    return y

def Q_l_fit(train_traj, f_target="sum", max_deg = 1, n_tilde = 100):
    N_train = train_traj.shape[0]
    N = train_traj.shape[1]
    d = train_traj.shape[2]
    Betas = np.zeros((N, d+ 1 + (max_deg-1) * int(d*(d+1)/2)))
    for l in tqdm(range(N)):
        # Linear Regression
        if 0 < max_deg < 6:
            X, degrees = generate_X_poly(train_traj, l, max_deg)
        else:
            raise Exception('max_deg should be 1 or 2')
        y = generate_y_sum(train_traj, l, f_target, n_tilde)

        beta = np.linalg.inv(X.T @ X) @ X.T @ y
        Betas[l] = beta
    return Betas, degrees

def a_lk(traj, traj_grad, l, k_vec, step, degrees, Betas):
    dim = traj.shape[1]
    S = 0
    x_hat = traj[l-1] - step/2 *traj_grad[l-1] 
    Small_s = np.zeros(dim)
    for ind,deg in enumerate(degrees):
        Small_s[:] = 0
        for d, i in enumerate(deg):
            for t in range (i+1):
                for s in range (int(t/2 +1)):
                    if (k_vec[d] == t - 2*s):
                        Small_s[d] = Small_s[d] + comb(N=i, k = t, exact = True) * x_hat[0]**(i-t) * math.factorial(t)*1/math.factorial(s)*1 / np.sqrt(math.factorial(t-2*s)) *np.sqrt(step)**t /2**s
                    else:
                        pass
        S = S + Betas[l,ind] * Small_s.prod()
    return S

def M_bias(k_vec, traj, traj_grad, traj_noise, step, degrees, Betas):
    N = traj.shape[0]
    S = 0
    for l in range (N):
        s = a_lk(traj,traj_grad,l, k_vec, step, degrees, Betas)* Hermite_val(k_vec,traj_noise[l])
        S = S + s
    return S

def estimator_bias(k_vec, test_traj, test_traj_grad, test_traj_noise, step, degrees, Betas, n_jobs = -1):
    N_test = test_traj.shape[0]
    M_results = Parallel(n_jobs = n_jobs)(delayed(M_bias)(k_vec, test_traj[i], test_traj_grad[i], test_traj_noise[i], step, degrees, Betas)for i in range(N_test))
    return np.array(M_results).reshape(-1)

