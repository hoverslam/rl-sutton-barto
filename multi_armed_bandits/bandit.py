import numpy as np

class Bandit:
    ''' A single slot machine that gives a normally distributed reward 
    with random mean and variance. '''
    
    def __init__(self):
        self.mean = 5 * np.random.rand()    # from 0 to 5
        self.variance = np.random.rand()    # from 0 to 1
        
    def get_reward(self):
        return self.variance * np.random.randn() + self.mean

class MultiArmedBandit:
    ''' A k-armed slot machine where every lever gives a normally distributed reward
    with random means and variances. '''    
    
    def __init__(self, k):
        self.k = k
        self.bandits = [Bandit() for _ in range(k)]
    
    def show_bandits(self):
        for i, b in enumerate(self.bandits):
            print("[{}] Mean: {:.2f}, Variance: {:.4f}".format(i, b.mean, b.variance))

    def get_reward(self, action):
        return self.bandits[action].get_reward()