import pandas as pd
import numpy as np

data = pd.read_csv('HARP_ROSCA_final.csv')

"""
the 4 treatments are:
    safe_box,
    locked_box, 
    health_pot, 
    health_savings 

The control is:
    encouragement

the outcome is: 
    fol2_illness_untreated_3mo
"""

# how many people did multitreat?
data['multitreat'].value_counts()

# drop the users that multitreated
# TODO Do we really want to throw away all the data from these users?
data = data[data['multitreat'] == 0]

# how many people in each treatment?
print(data['encouragement'].value_counts())
print(data['safe_box'].value_counts())
print(data['locked_box'].value_counts())
print(data['health_pot'].value_counts())
print(data['health_savings'].value_counts())


# count how many people got a positive outcome in the the different treatments
print('\n')
print('health_savings', data[data['health_savings'] == 1]['fol2_illness_untreated_3mo'].sum())
print('health_pot', data[data['health_pot'] == 1]['fol2_illness_untreated_3mo'].sum())
print('locked_box', data[data['locked_box'] == 1]['fol2_illness_untreated_3mo'].sum())
print('safe_box', data[data['safe_box'] == 1]['fol2_illness_untreated_3mo'].sum())

#count people treated with each treatment
print('\n')
print('health_savings', data[data['health_savings'] == 1].shape[0])
print('health_pot', data[data['health_pot'] == 1].shape[0])
print('locked_box', data[data['locked_box'] == 1].shape[0])
print('safe_box', data[data['safe_box'] == 1].shape[0])
print('\n')

# print the ratio of people that got a positive outcome in the different treatments
print('\n')
print('health_savings', data[data['health_savings'] == 1]['fol2_illness_untreated_3mo'].sum()/data[data['health_savings'] == 1].shape[0])
print('health_pot', data[data['health_pot'] == 1]['fol2_illness_untreated_3mo'].sum()/data[data['health_pot'] == 1].shape[0])
print('locked_box', data[data['locked_box'] == 1]['fol2_illness_untreated_3mo'].sum()/data[data['locked_box'] == 1].shape[0])
print('safe_box', data[data['safe_box'] == 1]['fol2_illness_untreated_3mo'].sum()/data[data['safe_box'] == 1].shape[0])

"""
health_savings 0.18032786885245902
health_pot 0.2827586206896552           "health pot" is the best treatment
locked_box 0.2564102564102564
safe_box 0.18803418803418803
"""
global best_treatment
best_treatment = 'health_pot'

# calculate the ATE of the 4 treatments:
avg_Y_control = data['fol2_illness_untreated_3mo'][data['encouragement'] == 1].mean()
avg_Y_safe_box = data['fol2_illness_untreated_3mo'][data['safe_box'] == 1].mean()
avg_Y_locked_box = data['fol2_illness_untreated_3mo'][data['locked_box'] == 1].mean()
avg_Y_health_pot = data['fol2_illness_untreated_3mo'][data['health_pot'] == 1].mean()
avg_Y_health_savings = data['fol2_illness_untreated_3mo'][data['health_savings'] == 1].mean()

ATE_safe_box = avg_Y_safe_box - avg_Y_control
ATE_locked_box = avg_Y_locked_box - avg_Y_control
ATE_health_pot = avg_Y_health_pot - avg_Y_control
ATE_health_savings = avg_Y_health_savings - avg_Y_control

print('ATE_safe_box: ', ATE_safe_box)
print('ATE_locked_box: ', ATE_locked_box)
print('ATE_health_pot: ', ATE_health_pot)
print('ATE_health_savings: ', ATE_health_savings)

# the best one is health_savings
# (also, the only one in the paper significant at 5%)

epsilon = 0
n_people = 125   # simulate 2 RCTs with n_people each (+ control group)
n_simulations = 1000
success_count = 0

for _ in range(n_simulations):
    # select the 2 treatments randomly
    T1, T2 = np.random.choice(
        ['safe_box', 'locked_box', 'health_pot', 'health_savings'], 2, replace=True)
    avg_Y_control = data['fol2_illness_untreated_3mo'][data['encouragement'] == 1].sample(
        n_people, replace=True).mean()  # TODO should we ignore control?
    avg_Y_T1 = data['fol2_illness_untreated_3mo'][data[T1]
                                                  == 1].sample(n_people, replace=True).mean()
    avg_Y_T2 = data['fol2_illness_untreated_3mo'][data[T2]
                                                  == 1].sample(n_people, replace=True).mean()

    # get the ATEs of the 2 treatments by sampling n_people people from each treatment
    ATE_T1 = avg_Y_T1 - avg_Y_control
    ATE_T2 = avg_Y_T2 - avg_Y_control

    # did we find the best treatment?
    # TODO if you have epsilon > 0, change the condition for success
    assert epsilon == 0
    if T1 == best_treatment:
        if ATE_T1 > ATE_T2:
            success_count += 1
    elif T2 == best_treatment:
        if ATE_T2 > ATE_T1:
            success_count += 1
print('\n')
print('success rate on RCT: ', success_count/n_simulations)
print('total people used: ',  n_people*2,  "for treatment and ", n_people, "for control " ) # 3 because we have 2 treatments and the control
print('\n')

# This function takes the alphas and betas of M  beta distributions and epsilon.
# returns the probabilities

def probability_eps_optimal(alpha,
                            beta,
                            epsilon,
                            list_treatments=["safe_box",
                                             "locked_box",
                                             "health_pot",
                                             "health_savings"]):
    """
    calculate the probability of being an eps-optimal treatment
    for each of the M beta distributions

    :param alpha: list of alphas of the beta distributions B(alpha, beta)
    :param beta: list of betas of the beta distributions B(alpha, beta)
    :param epsilon: the epsilon value
    :param list_treatments: list of the treatments
    :return: list of probabilities of being an eps-optimal treatment
    """
    # we will do this by montecarlo simulation. Sample 1000 times from each beta distribution
    # and calculate the probability of being an eps-optimal treatment

    # sample 1000 times from each beta distribution
    # create a pandas dataframe of the samples
    thetas = pd.DataFrame()
    for treatment in list_treatments:
        # sample 1000 thetas from the beta distribution
        thetas[treatment] = np.random.beta(
            alpha[treatment], beta[treatment], 1000)

    # calculate the probability of being an eps-optimal treatment,
    # that is, what is the probability that theta is the highest among all thetas or epsilon close to the highest

    # for each treatment, calculate the probability of being an eps-optimal treatment
    prob_eps_optimal = {}
    for treatment in list_treatments:
        # calculate the probability of being an eps-optimal treatment
        prob_eps_optimal[treatment] = (
            thetas[treatment] >= thetas.max(axis=1) - epsilon).mean() # TODO this does not seem to do what you think it does!!
    return prob_eps_optimal

n_simulations = 100
success_count = 0
for sim in range(n_simulations):

    # now try with multiarmed bandits:
    # The prior is a Uniform, i.e. a beta(1,1) distribution
    # The posterior is a beta distribution, and it is updated with the data
    # The posterior beta(alpha, beta) is explicitly determined by alpha and beta, which are
    # the number of successes plus 1 and failures plus 1, respectively.
    n_people_multiarmed = int(50)  # TODO get this number from the paper
    dict_alpha = {'safe_box': 1, 'locked_box': 1,
                  'health_pot': 1, 'health_savings': 1}
    dict_beta = {'safe_box': 1, 'locked_box': 1,
                 'health_pot': 1, 'health_savings': 1}
    # epsilon = 0 # TODO get a higher number. But then you have to change the condition of success. 
                # TODO the difference between the best and the second best in the paper is 0.06 
                # TODO the average outcomes that I got where: 
                                                                # 0.31 (control)
                                                                # 0.20
                                                                # 0.27 (second best)
                                                                # 0.32 (BEST)
                                                                # 0.19


    total_people_used = 0
    # round one:
    for treatment in ['safe_box', 'locked_box', 'health_pot', 'health_savings']:
        # sample n_people from each treatment with replacement 
        subsample_data = data[data[treatment] == 1].sample(n_people_multiarmed, replace=True)
        total_people_used += n_people_multiarmed
        # update the posterior
        dict_alpha[treatment] += subsample_data['fol2_illness_untreated_3mo'].sum()
        dict_beta[treatment] += n_people_multiarmed - \
            subsample_data['fol2_illness_untreated_3mo'].sum()

    # round two:
    # TODO: am I ignoring the people already sampled? no, I'm not. I think that you are taking them into account by using the posterior

    # choose the 2 treatments with the highest probability of being epsilon-optimal:
    # get the probabilities of being an eps-optimal treatment
    # TODO is this a good rule? should I get rid off half the treatments? 
    prob_eps_optimal = probability_eps_optimal(dict_alpha, dict_beta, epsilon)
    # print(prob_eps_optimal)

    # get the 2 treatments with the highest probability of being an eps-optimal treatment
    T1, T2 = sorted(prob_eps_optimal,
                    key=prob_eps_optimal.get, reverse=True)[:2]
    if T1 != best_treatment and T2 != best_treatment:
        continue

    # update the epsilon and the number of people to sample TODO 
    # epsilon =  ?
    # n_people_multiarmed = ?

    for treatment in [T1, T2]:
        # sample n_people from each treatment
        subsample_data = data[data[treatment] == 1].sample(n_people_multiarmed, replace=True)
        total_people_used += n_people_multiarmed

        # update the posterior
        dict_alpha[treatment] += subsample_data['fol2_illness_untreated_3mo'].sum()
        dict_beta[treatment] += n_people_multiarmed - \
            subsample_data['fol2_illness_untreated_3mo'].sum()

    prob_eps_optimal = probability_eps_optimal(dict_alpha, dict_beta, epsilon, [T1, T2])
    # get the best treatment
    estimated_best_treatment = sorted(prob_eps_optimal,
                            key=prob_eps_optimal.get, reverse=True)[0]                  
    if estimated_best_treatment == best_treatment:
        success_count += 1
    # print('\n')
    # print('alpha: ', dict_alpha)
    # print('beta: ', dict_beta)


print('success rate on multiarmed bandits: ', success_count/n_simulations)
print('total people used: ', total_people_used)


# TODO IT SEEMS THAT THE FUNCTION probability_eps_optimal DOES NOT DO WHAT YOU THINK IT DOES.
# TODO TEST IT! 

