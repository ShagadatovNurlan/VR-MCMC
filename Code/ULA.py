import numpy as np
from matplotlib import pyplot as plt

def ULA_with_burnin(d, step, burn_in, n, f_grad):
    """ MCMC ULA
    Args:
    	d: dimension
        step: stepsize of the algorithm
        burn_in: burn-in period
        n: number of samples after the burn-in
        f_grad: gradient of the potential U
    Returns:
        traj: a numpy array of size (n, d), where the trajectory is stored
        traj_grad: numpy array of size (n, d), where the gradients of the potential U along the trajectory are stored
        traj_noise: numpy array of size (n, d), where the noises along trajectory are stored
    """
    traj = np.zeros((burn_in + n, d))
    traj_grad = np.zeros((burn_in + n, d))
    traj_noise = np.random.randn(burn_in + n, d)
    
    traj[0] = (np.random.normal(0,1,d)).reshape(d)
    traj_grad[0] = f_grad(traj[0])

    for i in range(1,burn_in + n):
        traj[i] = traj[i-1] - step/2*traj_grad[i-1] + np.sqrt(step) * traj_noise[i]
        traj_grad[i] = f_grad(traj[i])
    return traj[burn_in:], traj_grad[burn_in:], traj_noise[burn_in:]

def ULA_from_initial(d, step, x_initial, n, f_grad):
	""" MCMC ULA (generate train trajectory from initial point)
    Args:
    	d: dimension
        step: stepsize of the algorithm
        x_initial: starting point
        n: number of samples after the burn-in
        f_grad: gradient of the potential U
    Returns:
        traj: a numpy array of size (n, d), where the trajectory is stored
        traj_grad: numpy array of size (n, d), where the gradients of the potential U along the trajectory are stored
        traj_noise: numpy array of size (n, d), where the noises along trajectory are stored
    """
	traj = np.zeros((n, d))
	traj_grad = np.zeros((n, d))
	traj_noise = np.random.randn(n, d)
	traj[0] = x_initial
	traj_grad[0] = f_grad(traj[0])

	for i in range(1, n):
		traj[i] = traj[i-1] - step/2*traj_grad[i-1] + np.sqrt(step) * traj_noise[i]
		traj_grad[i] = f_grad(traj[i])
	return traj, traj_grad, traj_noise

def generate_train_trajectories(sampling_trajectory, N_train, d, step, n, f_grad):
	train_traj = []
	train_traj_noise = []
	train_traj_grad = []
	for i in range(N_train):
		x_initial = sampling_trajectory[np.random.choice(np.arange(sampling_trajectory.shape[0]))]
		X, G, Z = ULA_from_initial(d, step, x_initial, n, f_grad)
		train_traj.append(X)
		train_traj_grad.append(G)
		train_traj_noise.append(Z)
	return np.array(train_traj),np.array(train_traj_grad), np.array(train_traj_noise)

def generate_test_trajetories(N_test, d, step, burn_in, n, f_grad):
 	test_traj = []
 	test_traj_noise = []
 	test_traj_grad = []
 	for i in range(N_test):
 		X, G, Z = ULA_with_burnin(d, step, burn_in, n, f_grad)
 		test_traj.append(X)
 		test_traj_grad.append(G)
 		test_traj_noise.append(Z)
 	return np.array(test_traj),np.array(test_traj_grad), np.array(test_traj_noise)

def plot_distr(traj, traj_noise, traj_grad, i):
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
    axs[0].hist(traj[:,i].reshape(-1,1), 45, density=True, facecolor='g', alpha=0.75)
    axs[0].set_title('trajectory')
    axs[1].hist(traj_noise[:,i].reshape(-1,1), 45, density=True, facecolor='g', alpha=0.75)
    axs[1].set_title('trajectory_noise')
    axs[2].hist(traj_grad[:,i].reshape(-1,1), 45, density=True, facecolor='g', alpha=0.75)
    axs[2].set_title('trajectory_grad')
    fig.tight_layout()
    plt.show()