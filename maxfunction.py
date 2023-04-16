import numpy as np
from scipy.stats import binom, norm
import matplotlib.pyplot as plt

def likelihood(alpha, data):
    return np.prod(binom.pmf(data, n=10, p=alpha))

def generate_data(alpha, size):
    return np.random.binomial(10, alpha, size)

def estimate_alpha(data):
    likelihoods = []
    alphas = np.linspace(0, 1, 1000)
    for alpha in alphas:
        likelihoods.append(likelihood(alpha, data))
    likelihoods = np.array(likelihoods)
    max_index = np.argmax(likelihoods)
    alpha_hat = alphas[max_index]
    # compute 95% confidence interval
    lower_bound = alphas[np.argmin((likelihoods - likelihoods[max_index] + np.log(0.05))**2)]
    upper_bound = alphas[np.argmin((likelihoods - likelihoods[max_index] + np.log(0.05))**2)]
    return alpha_hat, lower_bound, upper_bound

alpha_true = 0.7
num_experiments = 1000
num_measurements = 10

alpha_hats = []
lower_bounds = []
upper_bounds = []

for i in range(num_experiments):
    data = generate_data(alpha_true, num_measurements)
    alpha_hat, lower_bound, upper_bound = estimate_alpha(data)
    alpha_hats.append(alpha_hat)
    lower_bounds.append(lower_bound)
    upper_bounds.append(upper_bound)

print("Estimated alpha: ", np.mean(alpha_hats))
print("Confidence interval: [", np.mean(lower_bounds), ",", np.mean(upper_bounds), "]")

# plot the results
plt.hist(alpha_hats, bins=30, density=True, alpha=0.5, label='Estimated alpha')
plt.axvline(x=alpha_true, color='red', label='True alpha')
plt.xlabel('Alpha')
plt.ylabel('Density')
plt.title('Estimation of Alpha using Binomial Distribution')
plt.legend()
plt.show()
