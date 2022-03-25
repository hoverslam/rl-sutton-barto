# Evaluate a given policy

import numpy as np
import gridworlds

def policy_evaluation(pi, env, gamma=1, theta=10e-5):
    prev_V = np.zeros(len(env.state_space()), dtype=np.float64)
    while True:
        V = np.zeros(len(env.state_space()), dtype=np.float64)
        for s in range(len(env.state_space())):         
            for a, prob_a in enumerate(pi[s]):
                for prob_s, next_state, reward, done in env.step(s, a):
                    V[s] += prob_a * prob_s * (reward + gamma * prev_V[next_state] * (not done))                    
        if np.max(np.abs(prev_V - V)) < theta:
            break
        else:      
            prev_V = V.copy()
        
    return V        

env = gridworlds.Grid_4x4()
pi = np.ones([len(env.state_space()), len(env.action_space())]) / len(env.action_space())   # equiprobable random policy
V = policy_evaluation(pi, env)
env.print_values(V)