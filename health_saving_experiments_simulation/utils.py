import numpy as np 
import pandas as pd
import random

# This function takes the alphas and betas of M  beta distributions and epsilon.
# returns the probabilities

def probability_eps_optimal(alpha,
                            beta,
                            epsilon,
                            list_treatments,
                            n_sample = 1000
                            ):
    """
    calculate the probability of being an eps-optimal treatment
    for each of the M beta distributions

    :param alpha: list of alphas of the beta distributions B(alpha, beta)
    :param beta: list of betas of the beta distributions B(alpha, beta)
    :param epsilon: the epsilon value
    :param list_treatments: list of the treatments
    :return: list of probabilities of being an eps-optimal treatment
    """
    # we will do this by montecarlo simulation. 
    # Sample n_sample times from each beta distribution
    # and calculate the probability of being an eps-optimal treatment

    # sample n_sample times from each beta distribution
    # create a pandas dataframe of the samples
    thetas = pd.DataFrame()
    for treatment in list_treatments:
        # sample n_sample thetas from the beta distribution
        thetas[treatment] = np.random.beta(
            alpha[treatment], beta[treatment], n_sample) 

    # calculate the probability of being an eps-optimal treatment,
    # that is, what is the probability that theta is 
    # the highest among all thetas, or epsilon close to the highest

    # for each treatment, calculate the probability of being eps-optimal
    prob_eps_optimal = {}
    for treatment in list_treatments:
        # calculate the probability of being an eps-optimal treatment
        prob_eps_optimal[treatment] = (
            thetas[treatment] >= thetas.max(axis=1) - epsilon).mean() 
    return prob_eps_optimal