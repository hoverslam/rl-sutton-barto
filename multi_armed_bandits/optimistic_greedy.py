import numpy as np
import bandit

def select_action(rewards, epsilon):
    k = len(rewards)
    if np.random.rand() < epsilon:
        return np.random.randint(0, k)
    else:   
        best_actions = np.flatnonzero(rewards == np.max(rewards))
        return np.random.choice(best_actions)

# Parameters
steps = 1000
k = 10
epsilon = 0
initial_values = 5

# Optimistic-greedy algorithm
# Only differences to epsilon-greedy are the initial values of Q(a) and an epsilon of 0.
mab = bandit.MultiArmedBandit(k)
rewards = np.full(k, initial_values, dtype=np.float32)
taken = np.zeros(k, dtype=np.int32)

for _ in range(steps):
    action = select_action(rewards, epsilon)
    reward = mab.get_reward(action)
    taken[action] += 1
    rewards[action] = rewards[action] + (1 / taken[action]) * (reward - rewards[action])

# Results
print("###")
print("Best lever: [{}] with an estimated reward of {:.4f}".format(np.argmax(rewards), np.max(rewards)))
print("###")  
print("True values:")
mab.show_bandits()