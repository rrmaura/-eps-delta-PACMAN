# TODO: comments about table 3
# TODO: this gives epsilon, n and returns delta.
# Repeat program the other way around.

import pandas as pd
import numpy as np
import random
from utils import probability_eps_optimal
# data = pd.read_csv('/Users/sreeayyar/Dropbox/GitHub/PACMAN/health_saving_experiments_simulation/HARP_ROSCA_data.csv')
data = pd.read_csv('D:\GitHub Repositories\PACMAN\health_saving_experiments_simulation\HARP_ROSCA_data.csv')
data = data[data["has_followup2"] == 1]
# drop the users that multitreated
# TODO Do we really want to throw away all the data from these users? 
# - I think so; it is just over 10% of the sample.
# data = data[data['multitreat'] == 0]

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

treatments_and_control = ['health_savings', 
                'health_pot',
                'locked_box',
                'safe_box',
                'encouragement']



print("\n\nNumber of people allocated in each treatment?\n")
for t in treatments_and_control:
    print(t, data[data[t] == 1].shape[0])

print("\n\nNumber of people with positive outcomes in each treatment group\n")
for t in treatments_and_control:
    print(t, data[data[t] == 1]['fol2_illness_untreated_3mo'].sum())

print("\n\nRatio of positive outcomes to treated\n")
for t in treatments_and_control:
    print(t, round(data[data[t] == 1]['fol2_illness_untreated_3mo'].sum() / data[data[t] == 1].shape[0], 2))

"""
health_savings 0.180
health_pot 0.282        "health pot" is the best treatment
locked_box 0.256
safe_box 0.188
"""

# Calculate the ATE of the 4 treatments:
avg_Y_control = data['fol2_illness_untreated_3mo'][data['encouragement'] == 1].mean()
ATEs = {}
print("\n\nATEs\n")
for t in treatments_and_control[:-1]:
    avg_Y_treatment = data['fol2_illness_untreated_3mo'][data[t] == 1].mean()
    ATE = avg_Y_treatment - avg_Y_control
    ATEs[t] = ATE
    print(f'ATE_{t}: {round(ATE, 2)}')

global best_treatment
best_treatment = max(ATEs, key=ATEs.get)
print(f"\nBest Treatment: {best_treatment}")

# the best one is health_savings
# TODO: R: also, the only one in the paper significant at 5% - 
    # S: no. which table are you looking at? 
    # imo reference should be Table 3, columns (3) and (4)
    # should we be replicating their results for credibility? 
    # what about CIA-based arguments? (Y(1),Y(0) ind T | X)
    # R: Sorry, my bad, I agree with you that it is weird that in their results 
    # they got health savings while we got health pot.
    # I suspect that it is because they control for a lot of stuff that we don't.
    # also, they do not throw away the people that multitreated.
    # and they focus on people with follow up

epsilon = 0
n_people = 125   # simulate 2 RCTs with n_people each (+ control group)
n_simulations = 200
success_count = 0
randomtreat = int(1) # set to 1 if you want horserace to be against random two RCT treatments

for _ in range(n_simulations):
    # select the 2 treatments randomly
    if randomtreat == 1:
        T1, T2 = np.random.choice(
            ['safe_box', 'locked_box', 'health_pot', 'health_savings'], 2, replace=True)
    if randomtreat == 0:
        T1 = 'health_pot'
        T2 = 'safe_box'
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


n_simulations = 2000 
success_count = 0
for sim in range(n_simulations):

    # now try with multiarmed bandits:
    # The prior is a Uniform, i.e. a beta(1,1) distribution
    # The posterior is a beta distribution, and it is updated with the data
    # The posterior beta(alpha, beta) is determined by alpha and beta, which are
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
    # choose the 2 treatments with the highest probability of being epsilon-optimal:
    # get the probabilities of being an eps-optimal treatment
    prob_eps_optimal = probability_eps_optimal(dict_alpha, 
                                                dict_beta, 
                                                epsilon, 
                                                treatments_and_control[:-1])

    # get the 2 treatments with the highest probability of being an eps-optimal treatment
    T1, T2 = sorted(prob_eps_optimal,
                    key=prob_eps_optimal.get, reverse=True)[:2]
    if T1 != best_treatment and T2 != best_treatment:
        continue

    # TODO: update the epsilon and the number of people to sample?
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
    # print('beta: ', dict_beta)`

print('success rate on multiarmed bandits: ', success_count/n_simulations)
print('total people used: ', total_people_used)

# TODO: do not randomly select RCT treatments, run it against the "best" ATE treatment/base it off paper hypothesis/horserace against any 2 treatments - 
    # it seems this randomness is driving our bandits being better? - if economists know what to do, then all good?
# TODO: try different epsilons (0.05 is high for a binary variable; let us go 0.01) - normative judgement (plot graph)

# %% Tests of the probability_eps_optimal function:

# TODO: IT SEEMS THAT THE FUNCTION probability_eps_optimal DOES NOT DO WHAT YOU THINK IT DOES - in what sense??

random.seed(17)

scores = pd.DataFrame()

scores[1] = [97,77,99,77,17]
scores[2] = [18,188,20,14,12]
scores[3] = [89,55,12,59,23]

prob_eps_optimal = (scores[1] >= scores.max(axis=1) - epsilon).mean()
prob_eps_optimal

scores = pd.DataFrame()

scores[1] = np.random.beta(1, 1, 20) 
scores[2] = np.random.beta(1, 1, 20) 
scores[3] = np.random.beta(1, 1, 20) 
scores

prob_eps_optimal = (scores[1] >= scores.max(axis=1) - epsilon).mean()
prob_eps_optimal

