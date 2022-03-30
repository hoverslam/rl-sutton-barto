# Finds the best policy for a given MDP using value iteration

import numpy as np
import gridworlds

def update_policy(actions, env):
    pi = np.zeros([len(actions), len(env.action_space())], dtype=np.int32)
    for i in range(len(actions)):
        pi[i][actions[i]] = 1
        
    return pi

def value_iteration(env, gamma=1.0, theta=10e-5):
    V = np.zeros(len(env.state_space()), dtype=np.float64)
    i = 0
    while True:
        Q = np.zeros([len(env.state_space()), len(env.action_space())], dtype=np.float64)
        for s in range(len(env.state_space())):         
            for a in range(len(env.action_space())):
                for prob, next_state, reward, done in env.step(s, a):
                    Q[s][a] += prob * (reward + gamma * V[next_state] * (not done))                    
        if np.max(np.abs(V - np.max(Q, axis=1))) < theta:
            break
    
        V = np.max(Q, axis=1)
        i += 1
        
    pi = update_policy(np.argmax(Q, axis=1), env)
            
    return V, pi, i

env = gridworlds.Grid_5x5_Sutton()
V, pi, i = value_iteration(env, 0.9)
print("Optimal policy found in {} iterations:".format(i))
env.print_policy(pi) 
print("State values for optimal policy:")
env.print_values(V)
