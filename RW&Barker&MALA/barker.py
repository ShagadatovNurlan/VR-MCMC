import numpy as np
from matplotlib import pyplot as plt

def RWM_with_burnin(d, step, burn_in, n, f):
    traj = np.zeros((burn_in + n, d))
    traj_noise_g = np.random.randn(n, d)
    traj_noise_u = np.random.uniform(size = n)
    x = np.random.randn(d).reshape(d)

    for k in np.arange(burn_in):
        y = x + np.sqrt(step) * np.random.normal(size=d)
        logratio = -f(y) + f(x)
        if np.log(np.random.uniform()) <= logratio:
            x = y

    for k in np.arange(n):
        traj[k,] = x
        y = x + np.sqrt(step) * traj_noise_g[k]
        logratio = -f(y) + f(x)
        if np.log(traj_noise_u[k]) <= logratio:
            traj[k] = y
    return (traj, traj_noise_g, traj_noise_u)

def RWM_from_initial(d, step, x_initial, n, f):

    traj = np.zeros((n, d))
    traj_noise_g = np.random.randn(n, d)
    traj_noise_u = np.random.uniform(size = n)
    x = x_initial

    for k in np.arange(n):
        traj[k,] = x
        y = x + np.sqrt(step) * traj_noise_g[k]
        logratio = -f(y) + f(x)
        if np.log(traj_noise_u[k]) <= logratio:
            traj[k] = y
    return (traj, traj_noise_g, traj_noise_u)

def generate_train_trajectories(sampling_trajectory, N_train, d, step, n, f):
    train_traj = []
    train_traj_noise_g = []
    train_traj_noise_u = []
    for i in range(N_train):
        x_initial = sampling_trajectory[np.random.choice(np.arange(sampling_trajectory.shape[0]))]
        X, G, U = RWM_from_initial(d, step, x_initial, n, f)
        train_traj.append(X)
        train_traj_noise_g.append(G)
        train_traj_noise_u.append(U)
    return np.array(train_traj),np.array(train_traj_noise_g), np.array(train_traj_noise_u)

def generate_test_trajetories(N_test, d, step, burn_in, n, f):
    test_traj = []
    test_traj_noise_g = []
    test_traj_noise_u = []
    for i in range(N_test):
        X, G, U = RWM_with_burnin(d, step, x_initial, n, f)
        test_traj.append(X)
        test_traj_noise_g.append(G)
        test_traj_noise_u.append(U)
    return np.array(test_traj),np.array(test_traj_noise_g), np.array(test_traj_noise_u)

def plot_distr(traj, traj_noise_g, traj_noise_u, i):
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
    axs[0].hist(traj[:,i].reshape(-1,1), 45, density=True, facecolor='g', alpha=0.75)
    axs[0].set_title('trajectory')
    axs[1].hist(traj_noise_g[:,i].reshape(-1,1), 45, density=True, facecolor='g', alpha=0.75)
    axs[1].set_title('trajectory_noise_g')
    axs[2].hist(traj_noise_u.reshape(-1,1), 45, density=True, facecolor='g', alpha=0.75)
    axs[2].set_title('trajectory_noise_u')
    fig.tight_layout()
    plt.show()
