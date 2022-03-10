import numpy as np

class Bandit:
    ''' A single slot machine that gives a normally distributed reward 
    with random mean and unit variance. '''
    
    def __init__(self):
        self.mu = np.random.randn()   # drawn from a standard normal distribution
        self.sigma = 1
        
    def get_reward(self):
        return np.random.normal(self.mu, self.sigma)

class MultiArmedBandit:
    ''' A k-armed slot machine where every lever gives a different reward. '''    
    
    def __init__(self, k):
        self.k = k
        self.bandits = [Bandit() for _ in range(k)]
    
    def show_bandits(self):
        for i, b in enumerate(self.bandits):
            print("[{}] Mean: {:.4f}, Variance: {:.4f}".format(i, b.mu, b.sigma))

    def get_reward(self, action):
        return self.bandits[action].get_reward()