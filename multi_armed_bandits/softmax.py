import numpy as np
import bandit

# Softmax action selection
def softmax(x, temp):
    return np.exp(x / temp) / sum(np.exp(x / temp))

def select_action(rewards, temp):
    weights = softmax(rewards, temp)
    totals = np.cumsum(weights)
    norm = totals[-1]
    throw = np.random.rand() * norm
    
    return np.searchsorted(totals, throw)

# Parameters
steps = 1000
k = 10
temp = 0.2

# Training
mab = bandit.MultiArmedBandit(k)
rewards = np.zeros(k)
taken = np.zeros(k, dtype=np.int32)

for _ in range(steps):
    action = select_action(rewards, temp)
    reward = mab.get_reward(action)
    taken[action] += 1
    rewards[action] = rewards[action] + (1 / taken[action]) * (reward - rewards[action])

# Results
print("###")
print("Best lever: [{}] with an estimated reward of {:.4f}".format(np.argmax(rewards), np.max(rewards)))
print("###")  
print("True values:")
mab.show_bandits()