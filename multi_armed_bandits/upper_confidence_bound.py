import numpy as np
import bandit

# Upper-Confidence-Bound action selection
def select_action(rewards, taken, t, c):
    k = len(rewards)
    upper_confidence_bounds = np.zeros(k, dtype=np.float32)
    best_actions = np.where(taken == 0)[0]
    if len(best_actions) == 0:
        for i in range(k):
            upper_confidence_bounds[i] = rewards[i] + c * np.sqrt(np.log(t) / taken[i])
        best_actions = np.flatnonzero(upper_confidence_bounds == np.max(upper_confidence_bounds))
    action = np.random.choice(best_actions)    # break ties randomly
    
    return action

# Parameters
steps = 1000
k = 10
c = 2

# Training
mab = bandit.MultiArmedBandit(k)
rewards = np.zeros(k)
taken = np.zeros(k, dtype=np.int32)

for t in range(steps):
    action = select_action(rewards, taken, t, c)
    reward = mab.get_reward(action)
    taken[action] += 1
    rewards[action] = rewards[action] + (1 / taken[action]) * (reward - rewards[action])

# Results
print("###")
print("Best lever: [{}] with an estimated reward of {:.4f}".format(np.argmax(rewards), np.max(rewards)))
print("###")  
print("True values:")
mab.show_bandits()