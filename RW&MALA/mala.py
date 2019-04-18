import numpy as np
from matplotlib import pyplot as plt

def MALA_with_burnin(d, step, burn_in, n, f_grad, f):
    """ MCMC MALA
    Args:
        d: dimension
        step: stepsize of the algorithm
        burn_in: burn-in period
        n: number of samples after the burn-in
        f_grad: gradient of the potential U
    Returns:
        traj: a numpy array of size (n, d), where the trajectory is stored
        traj_grad: numpy array of size (n, d), where the gradients of the
            potential U along the trajectory are stored
    """
    traj = np.zeros((n, d))
    traj_grad = np.zeros((n, d))
    traj_noise_g = np.random.randn(n, d)    
    traj_noise_u = np.random.uniform(size = n)
    x = np.random.normal(size=d)
    
    for k in np.arange(burn_in):
        y = x - step/2 * f_grad(x) + np.sqrt(step)*np.random.normal(size=d)
        logratio = -f(y)+f(x) + (1./(2*step))*(np.linalg.norm(y-x+step/2*f_grad(x))**2 \
                      - np.linalg.norm(x-y+step/2*f_grad(y))**2)
        if np.log(np.random.uniform())<=logratio:
            x = y
    counter = 0.0
    for k in np.arange(n):
        traj[k,]=x
        traj_grad[k,]=f_grad(x)
        y = x - step * f_grad(x) + np.sqrt(2*step)*traj_noise_g[k]
        logratio = -f(y)+f(x) + (1./(2*step))*(np.linalg.norm(y-x+step/2*f_grad(x))**2 \
                      - np.linalg.norm(x-y+step/2*f_grad(y))**2)
        if np.log(traj_noise_u[k])<=logratio:
            counter +=1
            x = y
    return traj, traj_grad, traj_noise_g, traj_noise_u, counter/n


def MALA_from_initial(d, step, x_initial, n, f_grad, f):
    
    traj = np.zeros((n, d))
    traj_grad = np.zeros((n, d))
    traj_noise_g = np.random.randn(n, d)    
    traj_noise_u = np.random.uniform(size = n)
    x = x_initial

    for k in np.arange(n):
        traj[k,]=x
        traj_grad[k,]=f_grad(x)
        y = x - step * f_grad(x) + np.sqrt(2*step)*traj_noise_g[k]
        logratio = -f(y)+f(x) + (1./(2*step))*(np.linalg.norm(y-x+step/2*f_grad(x))**2 \
                      - np.linalg.norm(x-y+step/2*f_grad(y))**2)
        if np.log(traj_noise_u[k])<=logratio:
            x = y
    return traj, traj_grad, traj_noise_g, traj_noise_u

def generate_train_trajectories(sampling_trajectory, N_train, d, step, n, f_grad, f):
    train_traj = []
    train_traj_noise_g = []
    train_traj_noise_u = []
    train_traj_grad = []
    for i in range(N_train):
        x_initial = sampling_trajectory[np.random.choice(np.arange(sampling_trajectory.shape[0]))]
        X, G, Z, U = MALA_from_initial(d, step, x_initial, n, f_grad, f)
        train_traj.append(X)
        train_traj_grad.append(G)
        train_traj_noise_g.append(Z)
        train_traj_noise_u.append(U)
    return np.array(train_traj),np.array(train_traj_grad), np.array(train_traj_noise_g), np.array(train_traj_noise_u)

def generate_test_trajetories(N_test, d, step, burn_in, n, f_grad, f):
    test_traj = []
    test_traj_noise_g = []
    test_traj_noise_u = []
    test_traj_grad = []
    for i in range(N_test):
        X, G, Z, U, _= MALA_with_burnin(d, step, burn_in, n, f_grad, f)
        test_traj.append(X)
        test_traj_grad.append(G)
        test_traj_noise_g.append(Z)
        test_traj_noise_u.append(U)
    return np.array(test_traj),np.array(test_traj_grad), np.array(test_traj_noise_g), np.array(test_traj_noise_u)

def plot_distr(traj, traj_noise_g, traj_noise_u, i, pi_true = None):
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
    if pi_true is not None:
        axs[0].plot(np.arange(-5, 5, 0.1), np.array(list(map(pi_true, np.arange(-5, 5, 0.1)))))
    axs[0].hist(traj[:,i].reshape(-1,1), 45, density=True, facecolor='g', alpha=0.75)
    axs[0].set_title('trajectory')
    axs[1].hist(traj_noise_g[:,i].reshape(-1,1), 45, density=True, facecolor='g', alpha=0.75)
    axs[1].set_title('trajectory_noise_g')
    axs[2].hist(traj_noise_u.reshape(-1,1), 45, density=True, facecolor='g', alpha=0.75)
    axs[2].set_title('trajectory_noise_u')
    fig.tight_layout()
    plt.show()