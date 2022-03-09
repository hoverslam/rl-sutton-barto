import numpy as np

class Bandit:
    ''' A single slot machine that gives a normally distributed reward 
    with random mean and unit variance. '''
    
    def __init__(self):
        self.mean = np.random.randn()   # drawn from a standard normal distribution
        self.variance = 1
        
    def get_reward(self):
        return self.variance * np.random.randn() + self.mean

class MultiArmedBandit:
    ''' A k-armed slot machine where every lever gives a different reward. '''    
    
    def __init__(self, k):
        self.k = k
        self.bandits = [Bandit() for _ in range(k)]
    
    def show_bandits(self):
        for i, b in enumerate(self.bandits):
            print("[{}] Mean: {:.4f}, Variance: {:.4f}".format(i, b.mean, b.variance))

    def get_reward(self, action):
        return self.bandits[action].get_reward()