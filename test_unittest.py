from  health_saving_experiments_simulation.simulation_RCT_vs_PACMAN import probability_eps_optimal
import unittest   # The test framework
import numpy as np 

LIST_TREATMENTS=["safe_box",
                                             "locked_box",
                                             "health_pot",
                                             "health_savings"]
DICT_ALPHA = {'safe_box': 1, 'locked_box': 1,
                  'health_pot': 1, 'health_savings': 1}
DICT_BETA = {'safe_box': 1, 'locked_box': 1,
                 'health_pot': 1, 'health_savings': 1}


class Test_probability_eps_optimal(unittest.TestCase):
    def test_probability_eps_optimal_0(self):
        # when epsilon = 0, if all data is the same, 
        # the probability of being optimal is 0.25 
        epsilon = 0
        pr = probability_eps_optimal(DICT_ALPHA,
                            DICT_BETA,
                            epsilon,
                            LIST_TREATMENTS)
        for treatment in LIST_TREATMENTS:    
            assert np.isclose(pr[treatment], 0.25, rtol=1e-02)

        
    
    

if __name__ == '__main__':
    unittest.main()