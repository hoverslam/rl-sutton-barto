# Evaluate a given policy

import numpy as np
import gridworlds

def policy_evaluation(pi, env, gamma=1, theta=10e-5):
    prev_V = np.zeros(len(env.state_space()), dtype=np.float64)
    i = 0
    while True:
        V = np.zeros(len(env.state_space()), dtype=np.float64)
        for s in range(len(env.state_space())):         
            for a, prob_a in enumerate(pi[s]):
                for prob_s, next_state, reward, done in env.step(s, a):
                    V[s] += prob_a * prob_s * (reward + gamma * prev_V[next_state] * (not done))                    
        if np.max(np.abs(prev_V - V)) < theta:
            break
        
        env.reset()      
        prev_V = V.copy()
        i += 1
        
    return V, i     

env = gridworlds.Grid_5x5_Sutton()
pi = np.ones([len(env.state_space()), len(env.action_space())]) / len(env.action_space())   # equiprobable random policy
V, i = policy_evaluation(pi, env, 0.9)
print("Policy evaluated in {} iterations:".format(i))
env.print_values(V)