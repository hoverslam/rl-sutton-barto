# Finds the best policy for a given MDP using policy iteration

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

def policy_improvement(V, env, gamma=1.0):
    Q = np.zeros([len(env.state_space()), len(env.action_space())], dtype=np.float64)
    for s in range(len(env.state_space())):
        for a in range(len(env.action_space())):
            for prob, next_state, reward, done in env.step(s, a):
                Q[s][a] += prob * (reward + gamma * V[next_state] * (not done))                
    pi = update_policy(np.argmax(Q, axis=1), env)
    
    return pi

def policy_iteration(env, gamma=1.0, theta=1e-5):
    pi = np.ones([len(env.state_space()), len(env.action_space())]) / 4
    i = 0
    while True:
        old_pi = pi.copy()
        V = policy_evaluation(pi, env, gamma, theta)
        pi = policy_improvement(V[0], env, gamma)
        if np.array_equal(old_pi, pi):
            break
          
        i += 1
    
    return V[0], pi, i

def update_policy(actions, env):
    pi = np.zeros([len(actions), len(env.action_space())], dtype=np.int32)
    for i in range(len(actions)):
        pi[i][actions[i]] = 1
        
    return pi

env = gridworlds.Grid_5x5_Sutton()
V, pi, i = policy_iteration(env, 0.9)
print("Optimal policy found in {} iterations:".format(i))
env.print_policy(pi) 
print("State values for optimal policy:")
env.print_values(V)