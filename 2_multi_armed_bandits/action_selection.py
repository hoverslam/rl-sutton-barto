import numpy as np

# Epsilon-greedy
def epsilon_greedy(rewards, epsilon):
    k = len(rewards)
    if np.random.rand() < epsilon:
        return np.random.randint(0, k)
    else:   
        best_actions = np.flatnonzero(rewards == np.max(rewards))
        return np.random.choice(best_actions)    # break ties randomly
  
# Upper-Confidence-Bound  
def ucb(rewards, taken, t, c):
    k = len(rewards)
    upper_confidence_bounds = np.zeros(k, dtype=np.float32)
    best_actions = np.where(taken == 0)[0]
    if len(best_actions) == 0:
        for i in range(k):
            upper_confidence_bounds[i] = rewards[i] + c * np.sqrt(np.log(t) / taken[i])
        best_actions = np.flatnonzero(upper_confidence_bounds == np.max(upper_confidence_bounds))
    
    return np.random.choice(best_actions)    # break ties randomly

# Softmax
def softmax(rewards, temp):
    weights = np.exp(rewards / temp) / sum(np.exp(rewards / temp))    # boltzmann distribution
    totals = np.cumsum(weights)
    norm = totals[-1]
    throw = np.random.rand() * norm
    
    return np.searchsorted(totals, throw)