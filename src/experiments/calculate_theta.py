from scipy.optimize import fsolve
import numpy as np

# Define Kendall's tau equation for the Frank copula
def kendalls_tau_frank(alpha, tau_target):
    if alpha == 0:
        return 0
    term1 = 1 - (4 / alpha)
    term2 = (4 / alpha) * (1 - np.exp(-alpha)) / (alpha * (np.exp(-alpha) - 1))
    return tau_target - (term1 + term2)

# Solve for alpha given tau
tau = 0.25
alpha_initial_guess = 1  # Initial guess for alpha
alpha_solution = fsolve(kendalls_tau_frank, alpha_initial_guess, args=(tau,))
alpha = alpha_solution[0]

print(f"Alpha corresponding to Kendall's Tau = {tau}: {alpha:.4f}")