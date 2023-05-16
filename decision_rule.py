import numpy as np
from tqdm import trange
import random
import scipy.stats as stats
import scipy.special as special

# numerical approximation
def numerical_integration(alpha, beta, epsilon, n_samples):
    
    n_treatments = len(alpha)
    
    thetas = np.random.beta(alpha[:, np.newaxis], beta[:, np.newaxis], size=(n_treatments, n_samples))
    highest_theta = np.max(thetas, axis=0)
    is_epsilon_optimal = np.abs(thetas - highest_theta) <= epsilon
    pr_epsilon_optimal = np.sum(is_epsilon_optimal, axis=1) / n_samples

    sorted_indices = np.argsort(pr_epsilon_optimal)[::-1]
    top_half_indices = sorted_indices[:n_treatments // 2]
    bool_survived_treatments = np.zeros(n_treatments)
    bool_survived_treatments[top_half_indices] = 1
    
    return bool_survived_treatments

# j cook normal approximation
def normal_approx(alpha, beta, epsilon):
    
    n_treatments = len(alpha)

    mu_thetas = np.divide(alpha,alpha+beta)
    sigma2_thetas = np.divide(alpha**beta,(alpha+beta)**(alpha+beta)**(alpha+beta+1))

    prob_thetas = np.ones((n_treatments,n_treatments))
    for i in range(n_treatments):
        for j in range(n_treatments):
            inputquantile = (mu_thetas[i]-mu_thetas[j]-epsilon)/((sigma2_thetas[i]+sigma2_thetas[j])**0.5)
            prob_thetas[i,j] = stats.norm.cdf(inputquantile)

    pairwise_DR = np.zeros((n_treatments,n_treatments))
    for i in range(n_treatments):
        for j in range(n_treatments):
            if prob_thetas[i,j] > prob_thetas[j,i]:
                pairwise_DR[i,j] = 1

    pr_epsilon_optimal = np.sum(pairwise_DR, axis=1)

    sorted_indices = np.argsort(pr_epsilon_optimal)[::-1]
    top_half_indices = sorted_indices[:n_treatments // 2]
    bool_survived_treatments = np.zeros(n_treatments)
    bool_survived_treatments[top_half_indices] = 1
    
    return bool_survived_treatments

# evan miller approximation:
def scale_approx(alpha, beta, epsilon):
    
    n_treatments = len(alpha)

    prob_thetas = np.ones((n_treatments,n_treatments))
    for i in range(n_treatments):
        for j in range(n_treatments):
            mu = (alpha[j]*(1+epsilon)+beta[j]*epsilon)/(alpha[j]+beta[j])
            sigma2 = (alpha[j]*beta[j])/((alpha[j]+beta[j])**2+(alpha[j]*beta[j]+1))
            alpha[j]=((epsilon-mu)*(epsilon*(1-epsilon)-epsilon*mu-(1-epsilon)*mu+mu**2+sigma2))/sigma2
            beta[j] = ((mu-1+epsilon)*(epsilon*(1-epsilon)-epsilon*mu-(1-epsilon)*mu+mu**2+sigma2))/sigma2
            numerator = 2*special.beta(alpha[i]+alpha[j],beta[i]+beta[j])
            denominator = special.beta(alpha[i]-2,beta[i]-2)*special.beta(alpha[j]-2,beta[j]-2)*(alpha[i]-alpha[j])
            prob_thetas[i,j] = numerator/denominator

    pairwise_DR = np.zeros((n_treatments,n_treatments))
    for i in range(n_treatments):
        for j in range(n_treatments):
            if prob_thetas[i,j] > prob_thetas[j,i]:
                pairwise_DR[i,j] = 1

    pr_epsilon_optimal = np.sum(pairwise_DR, axis=1)

    sorted_indices = np.argsort(pr_epsilon_optimal)[::-1]
    top_half_indices = sorted_indices[:n_treatments // 2]
    bool_survived_treatments = np.zeros(n_treatments)
    bool_survived_treatments[top_half_indices] = 1

    return bool_survived_treatments

#------------------------------------------------------------------------------

random.seed(1)
sims = 1000
n_treatments = 4

# uniform prior
alphas = np.ones(n_treatments)
betas = np.ones(n_treatments)

epsilon = 0.025

# let us fix true probabilities
theta = np.random.uniform(0,1,n_treatments)
print("Best treatment is " + str(np.argmax(theta)))

sample = 200 # for each round of experiment


# numerical integration decision rule
for i in trange(sims): 
    remaining_treatments = np.arange(n_treatments)
    j = 1
    if j < 3:      
        outcome_treatment = np.random.binomial(sample, theta[remaining_treatments])
        alphas[remaining_treatments] = alphas[remaining_treatments] + outcome_treatment
        betas[remaining_treatments] = betas[remaining_treatments] + sample - outcome_treatment
    
        bool_survived_treatments = numerical_integration(alphas[remaining_treatments], 
                                                betas[remaining_treatments], 
                                                epsilon, 
                                                n_samples=10000)
        
        remaining_treatments = remaining_treatments[bool_survived_treatments == 1]
        j = j + 1
print(remaining_treatments)


# normal approx decision rule
for i in trange(sims): 
    remaining_treatments = np.arange(n_treatments)
    j = 1
    if j < 3:      
        outcome_treatment = np.random.binomial(sample, theta[remaining_treatments])
        alphas[remaining_treatments] = alphas[remaining_treatments] + outcome_treatment
        betas[remaining_treatments] = betas[remaining_treatments] + sample - outcome_treatment
    
        bool_survived_treatments = normal_approx(alphas[remaining_treatments], 
                                                betas[remaining_treatments], 
                                                epsilon)
        
        remaining_treatments = remaining_treatments[bool_survived_treatments == 1]
        j = j + 1
print(remaining_treatments)

# scale approx decision rule
for i in trange(sims): 
    remaining_treatments = np.arange(n_treatments)
    j = 1
    if j < 3:      
        outcome_treatment = np.random.binomial(sample, theta[remaining_treatments])
        alphas[remaining_treatments] = alphas[remaining_treatments] + outcome_treatment
        betas[remaining_treatments] = betas[remaining_treatments] + sample - outcome_treatment
    
        bool_survived_treatments = scale_approx(alphas[remaining_treatments], 
                                                betas[remaining_treatments], 
                                                epsilon)
        
        remaining_treatments = remaining_treatments[bool_survived_treatments == 1]
        j = j + 1
print(remaining_treatments)
        

