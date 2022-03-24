# Improving a random policy

import numpy as np
import gridworlds

def policy_evaluation(pi, P, gamma=1, theta=10e-5):
    prev_V = np.zeros(len(P))
    while True:
        V = np.zeros(len(P))
        for s in range(len(P)):            
            for prob_a, a in pi(s):
                for prob_s, next_state, reward, done in P[s][a]:
                    V[s] += prob_a * prob_s * (reward + gamma * prev_V[next_state] * (not done))    
        
        if np.max(np.abs(prev_V - V)) < theta:
            break      
        prev_V = V.copy()

    return V           

grid = gridworlds.Grid_4x4()
pi = lambda s : [(0.25, 0), (0.25, 1), (0.25, 2), (0.25, 3)]    # equiprobable random policy
V_pi = policy_evaluation(pi, grid.P)
grid.print_values(V_pi)