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
    X = np.empty((N_train * (N-r) ,d+1 + int(d*(d+1)/2)))
    all_points = train_traj[:, :N-r].reshape(-1,d)
    poly = PolynomialFeatures(max_deg)
    X = poly.fit_transform(all_points)
    return X, poly.powers_

def generate_y_sum(train_traj, l, n_tilde):
    N_train = train_traj.shape[0]
    N = train_traj.shape[1]
    d = train_traj.shape[2]
    y = np.zeros(N_train)
    for s in range(N_train):
        y[s] = train_traj[s][l:l+n_tilde].sum()
    return y

def Q_l_fit(train_traj, max_deg = 1, n_tilde = 5, f_target = 1, coord = None):
    N_train = train_traj.shape[0]
    N = train_traj.shape[1]
    d = train_traj.shape[2]
    Betas = np.zeros((N, d+ 1 + (max_deg-1) * int(d*(d+1)/2)))
    for l in tqdm(range(N)):
        # Linear Regression
        if 0 < max_deg < 3:
            X, degrees = generate_X_poly(train_traj, l, max_deg)
        else:
            raise Exception('max_deg should be 1 or 2')
        # if f_target == 1:
        #     y = generate_y_sum(train_traj, r)
        # if f_traget ==2 and coord is not None:
        #     y = generate_y_coord(train_traj, r, coord)
        # else:
        #     raise Exception('unknown target function')
        y = generate_y_sum(train_traj, l, n_tilde)

        beta = np.linalg.inv(X.T @ X) @ X.T @ y
        Betas[r] = beta
    return Betas, degrees