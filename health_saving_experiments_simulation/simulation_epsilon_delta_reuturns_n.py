# # given a set of epsilon_L and delta_L, given a prior, 
# # what is the minimum number of samples needed to
# # guarantee that the probability of being eps-optimal is at least 1-delta

# # TODO: work with numpy arrays instead of lists and dictionaries
# import numpy as np
# import scipy.stats as stats
# import matplotlib.pyplot as plt
# import math
# from utils import probability_eps_optimal

# def sample_complexity(epsilon_L, delta_L, alpha, beta, n_simulations=1000):
#     """
#     This funtion calculates what is the necessary amount of people necessary to 
#     guarantee that the probability of being eps-optimal is at least 1-delta_L
    
#     Parameters
#     ----------
#     epsilon_L : float
#         the epsilon_L value for the epsilon_L-optimal treatment
#     delta_L : float
#         the delta_L value for the epsilon_L-optimal treatment
#     alpha : numpy array
#         the alpha parameters of the Beta distribution
#     beta : numpy array
#         the beta parameters of the Beta distribution
#     treatments : numpy array
#         the treatments

#     n_simulations : int, optional
#         the number of simulations to run. The default is 1000.

#     Returns
#     -------
#     n : int
#         the minimum number of samples needed to guarantee that the 
#         probability of being eps-optimal is at least 1-delta_L
#     """
#     # TODO: ensure you deal with ties
#     # assume same number of samples for each treatment
#     # super simple approach. 
#     alpha_sim = np.repeat(alpha, n_simulations).reshape(n_treatments, n_simulations)
#     beta_sim = np.repeat(beta, n_simulations).reshape(n_treatments, n_simulations)
#     thetas = np.random.beta(alpha_sim, 
#                             beta_sim,
#                             (n_treatments, n_simulations))
    
#     # thetas.shape = (n_treatments, n_simulations)

#     highest_theta = np.max(thetas, axis=0)
#     is_epsilon_optimal = np.abs(thetas - highest_theta) <= epsilon_L

#     # for n in range(MAX_N): # alternatively do a binary search

#     left = 1
#     right = MAX_N
#     while left <= right:
#         n = (left + right) // 2

#         # sample n units of each treatment
#         # each treatment i follows a binomail distribution with 
#         # n trials and probability of success theta_i

#         # dim = n_treatments x n_simulations
#         sum_Y = np.random.binomial(n, thetas) 
        
        
#         updated_alpha = alpha_sim + sum_Y
#         updated_beta = beta_sim + n - sum_Y
                
#         bool_survived_treatments = bool_choice_rule(updated_alpha, 
#                                                     updated_beta, 
#                                                     epsilon_L)
        
#         # there is success if at least one of the treatments in the simulation 
#         # is epsilon optimal and has survived
#         success = is_epsilon_optimal * bool_survived_treatments
#         simulation_with_success = np.sum(success, axis=0) > 0
#         percentage_success =  np.sum(simulation_with_success) / n_simulations 
    

#         # if percentage_success > 1-delta_L:
#         #     return n

#         if percentage_success > 1-delta_L:
#             right = n - 1
#         else:
#             left = n + 1
        
#     return left


# def bool_choice_rule(alpha, beta, epsilon_L):
#     """
#     Given the alpha and beta parameters of the beta distributions,
#     return a np.array of boolean values indicating whether the treatment
#     should be kept or not. 

#     To do so, first calculate the probability that each treatment is
#     epsilon optimal. To do that, sample from the beta distribution
#     and calculate the amount of times that each treatment is epsilon optimal.

#     Then, for each simulation, keep the half of the treatments with highest
#     probability of being epsilon optimal.

#     Parameters
#     ----------
#     alpha : numpy array of size (n_treatments, n_simulations)
#         the alpha parameters of the Beta distribution
#     beta : numpy array of size (n_treatments, n_simulations)
#         the beta parameters of the Beta distribution
#     treatments : numpy array
#         the treatments 
#     epsilon_L : float
#         the epsilon_L value for the epsilon_L-optimal treatment

#     Returns
#     -------
#     bool_survived_treatments : numpy array of size (n_treatments, n_simulations)
#         a numpy array indicating whether the treatment should be kept or not. 
#         You want to keep the treatments that have highest probability
#         of being epsilon optimal
#     """
#     n_sample = alpha.shape[1]
#     n_sim = alpha.shape[1]

#     thetas = np.random.beta(np.repeat(alpha, n_sample).reshape((n_treatments, n_sim, n_sample)),
#                             np.repeat(beta, n_sample).reshape((n_treatments, n_sim, n_sample)), 
#                             (n_treatments, n_sim, n_sample))
#     # thetas.shape = (n_treatments, n_sim, n_sample)

#     highest_theta = np.max(thetas, axis=0)
#     # highest_theta.shape = (n_sim, n_sample)

#     is_epsilon_optimal = np.abs(thetas - highest_theta) <= epsilon_L
#     # is_epsilon_optimal.shape = (n_treatments, n_sim, n_sample)

#     pr_epsilon_optimal = np.sum(is_epsilon_optimal, axis=2) / n_sim

#     # Get the indices that would sort the pr_epsilon_optimal array along the first axis (treatments)
#     sorted_indices = np.argsort(pr_epsilon_optimal, axis=0)

#     # Reverse the order of indices (highest probability first)
#     sorted_indices = sorted_indices[::-1]

#     # Keep only the first half of the sorted indices
#     top_half_indices = sorted_indices[:n_treatments // 2]

#     # Create an array of shape (n_treatments, n_sim) filled with zeros
#     bool_survived_treatments = np.zeros((n_treatments, n_sim))

#     # Use advanced indexing to set the corresponding elements in bool_survived_treatments to 1
#     bool_survived_treatments[top_half_indices, np.arange(n_sim)] = 1

#     return bool_survived_treatments




# MAX_N = 100
# n_treatments=4
# # start with uniform priors, i.e. B(1,1)
# # i.e. we need the alpha and beta parameters of our Beta distribution to be 1
# alpha = np.ones(n_treatments)
# beta = np.ones(n_treatments)

# n = sample_complexity(epsilon_L=0.01,
#                     delta_L=0.01,
#                     alpha=alpha,
#                     beta=beta,
#                     n_simulations=10000)

# print(f"n = {n}")

# OPTIMIZED FASTER VERSION BY GPT4

import numpy as np
import math

def sample_complexity(epsilon_L, delta_L, alpha, beta, n_simulations=1000, n_samples=1000):
    n_treatments = len(alpha)
    MAX_N = 100

    left = 1
    right = MAX_N
    while left <= right:
        n = (left + right) // 2

        simulation_with_success = 0
        for _ in range(n_simulations):
            thetas = np.random.beta(alpha, beta, size=n_treatments)
            highest_theta = np.max(thetas)
            is_epsilon_optimal = np.abs(thetas - highest_theta) <= epsilon_L

            sum_Y = np.random.binomial(n, thetas)
            updated_alpha = alpha + sum_Y
            updated_beta = beta + n - sum_Y

            bool_survived_treatments = bool_choice_rule(updated_alpha, updated_beta, epsilon_L, n_samples)

            success = is_epsilon_optimal * bool_survived_treatments
            if np.sum(success) > 0:
                simulation_with_success += 1

        percentage_success = simulation_with_success / n_simulations

        if percentage_success > 1 - delta_L:
            right = n - 1
        else:
            left = n + 1

    return left

def bool_choice_rule(alpha, beta, epsilon_L, n_samples):
    n_treatments = len(alpha)
    thetas = np.random.beta(alpha[:, np.newaxis], beta[:, np.newaxis], size=(n_treatments, n_samples))
    highest_theta = np.max(thetas, axis=0)
    is_epsilon_optimal = np.abs(thetas - highest_theta) <= epsilon_L
    pr_epsilon_optimal = np.sum(is_epsilon_optimal, axis=1) / n_samples

    sorted_indices = np.argsort(pr_epsilon_optimal)[::-1]
    top_half_indices = sorted_indices[:n_treatments // 2]
    bool_survived_treatments = np.zeros(n_treatments)
    bool_survived_treatments[top_half_indices] = 1

    return bool_survived_treatments

n_treatments = 4
alpha = np.ones(n_treatments)
beta = np.ones(n_treatments)

n = sample_complexity(epsilon_L=0.01,
                      delta_L=0.01,
                      alpha=alpha,
                      beta=beta,
                      n_simulations=10000) # I get pretty unstable results with n=10e3, so I increased it to 10e4

print(f"n = {n}")
