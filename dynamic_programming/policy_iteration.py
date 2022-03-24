# Finds the best deterministic policy for a given MDP
# !!! Does not converge with gamma=1.0. Why?

import numpy as np
import gridworlds

def policy_evaluation(pi, P, gamma=1.0, theta=10e-5):
    prev_V = np.zeros(len(P))
    while True:
        V = np.zeros(len(P))
        for s in range(len(P)):            
                for prob, next_state, reward, done in P[s][pi[s]]:
                    V[s] += prob * (reward + gamma * prev_V[next_state] * (not done))    
        
        if np.max(np.abs(prev_V - V)) < theta:
            break      
        prev_V = V.copy()

    return V  

def policy_improvement(V, P, gamma=1.0):
    Q = np.zeros((len(P), len(P[0])), dtype=np.float64)
    for s in range(len(P)):
        for a in range(len(P[s])):
            for prob, next_state, reward, done in P[s][a]:
                Q[s][a] += prob * (reward + gamma * V[next_state] * (not done))                
    pi = np.argmax(Q, axis=1)
    
    return pi

def policy_iteration(P, gamma=1.0, theta=1e-5):
    action_space = tuple(P[0].keys())
    pi = np.random.choice(action_space, len(P))
    while True:
        old_pi = pi.copy()
        V = policy_evaluation(pi, P, gamma, theta)
        pi = policy_improvement(V, P, gamma)
        if np.array_equal(old_pi, pi):
            break
    
    return V, pi

grid = gridworlds.Grid_4x4()
V, pi = policy_iteration(grid.P, 0.9)
grid.print_policy(pi)
grid.print_values(V)