# Replicate the regression results from table 3 in the paper 
# Why Don’t the Poor Save More?
# Evidence from Health Savings Experiments†
# By Pascaline Dupas and Jonathan Robinson*

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns

# Read the data

data = pd.read_csv('D:\GitHub Repositories\PACMAN\health_saving_experiments_simulation\HARP_ROSCA_data.csv')


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
data = data[data["has_followup2"] == 1]

# Create interaction variable
data['bg_female_married'] = data['bg_female'] * data['bg_married']

# Define individual controls
individual_controls = "bg_b1_age bg_female bg_female_married bg_provider bg_hyperbolic bg_pat_now_impat_later bg_max_discount bg_n_roscas "

# substitute the space in individual controls with +
individual_controls = individual_controls.replace(" ", " + ")

strata_dummies = pd.get_dummies(data['strata'], prefix='strata')
data = pd.concat([data, strata_dummies], axis=1)

# Column 3
model = sm.OLS.from_formula("fol2_illness_untreated_3mo ~ safe_box + "\
                            "locked_box + health_pot + health_savings +"
                            " multitreat + rosbg_monthly_contrib + " + ' + '.join(strata_dummies.columns), data=data)




result = model.fit(cov_type='cluster', cov_kwds={'groups': data['id_harp_rosca']})
print(result.summary())
print(result.t_test(['safe_box']).summary())
print(result.t_test(['health_savings - safe_box']).summary())
# print(data.loc[(data['encouragement'] == 1) & (data['e_sample'] == 1), 'fol2_illness_untreated_3mo'].describe())

print(data.loc[(data['encouragement'] == 1), 'fol2_illness_untreated_3mo'].describe())

# Column 4
model = sm.OLS.from_formula("fol2_illness_untreated_3mo ~ safe_box + "\
                            "locked_box + health_pot + health_savings + "\
                                "multitreat + rosbg_monthly_contrib + " + individual_controls + ' + '.join(strata_dummies.columns), data=data)
result = model.fit(cov_type='cluster', cov_kwds={'groups': data['id_harp_rosca']})
print(result.summary())
print(result.t_test(['safe_box']).summary())
print(result.t_test(['health_savings - safe_box']).summary())
# print(data.loc[(data['encouragement'] == 1) & (data['e_sample'] == 1), 'fol2_illness_untreated_3mo'].describe())
print(data.loc[(data['encouragement'] == 1), 'fol2_illness_untreated_3mo'].describe())

# simplest model 
print("\n\n\n\n\n\n\n\n\n")

# our regression 

model = sm.OLS.from_formula("fol2_illness_untreated_3mo ~ safe_box + "\
                            "locked_box + health_pot + health_savings "\
                            , data=data)
result = model.fit(cov_type='cluster', cov_kwds={'groups': data['id_harp_rosca']})
print(result.summary())
print(result.t_test(['safe_box']).summary())
print(result.t_test(['health_savings - safe_box']).summary())