import numpy as np
from scipy.stats import norm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

def likelihood(alpha, data):
    """Calculate the likelihood of a parameter alpha given some data X"""
    mu = alpha  # assume a simple linear relationship between alpha and the mean of data
    sigma = 1   # assume a fixed standard deviation for simplicity
    return np.prod(norm.pdf(data, loc=mu, scale=sigma))

def simulate_experiment(alpha_true):
    """Simulate an experiment with a true parameter value alpha_true"""
    data = np.random.normal(loc=alpha_true, scale=1, size=10)  # assume 10 data points
    return data

def estimate_alpha(data):
    """Estimate the parameter alpha that maximizes the likelihood function"""
    res = minimize(lambda alpha: -likelihood(alpha, data), x0=0)
    return res.x[0]

def estimate_alpha_multiple_experiments(alpha_range, n_experiments):
    """Estimate the parameter alpha from multiple experiments with different true values"""
    alpha_estimates = []
    for i in range(n_experiments):
        data = simulate_experiment(alpha_true=alpha_range[i])
        alpha_estimate = estimate_alpha(data)
        alpha_estimates.append(alpha_estimate)
    return alpha_estimates

def confidence_interval(alpha_range, alpha_estimates, threshold=3.84):
    """Estimate the 95% confidence interval of the parameter alpha"""
    alpha_var = np.var(alpha_estimates, ddof=1)  # unbiased estimator of the variance
    log_likelihood_ratio = np.zeros_like(alpha_range)
    for i, alpha in enumerate(alpha_range):
        data = simulate_experiment(alpha_true=alpha)
        log_likelihood_ratio[i] = 2 * (likelihood(alpha, data) - likelihood(np.mean(alpha_estimates), data))
    alpha_ci = alpha_range[log_likelihood_ratio <= threshold * alpha_var]
    return alpha_ci

# simulate experiments with different values of alpha
alpha_range = np.linspace(-5, 5, 101)
n_experiments = 100
alpha_estimates = estimate_alpha_multiple_experiments(alpha_range, n_experiments)

# estimate the 95% confidence interval of alpha
alpha_ci = confidence_interval(alpha_range, alpha_estimates)

# plot the results in 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(alpha_range, np.zeros_like(alpha_range), alpha_estimates, c='b', marker='o')
ax.plot(alpha_range, np.zeros_like(alpha_range), alpha_range, c='r')
ax.plot(alpha_ci, np.zeros_like(alpha_ci), np.zeros_like(alpha_ci), c='g')
ax.set_xlabel('True alpha')
ax.set_ylabel('Y')
ax.set_zlabel('Estimated alpha')
plt.show()
