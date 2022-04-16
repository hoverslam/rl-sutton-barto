import numpy as np

# Policy evaluation: Sutton & Barto (2018), p. 75
def policy_evaluation(pi, env, gamma=1.0, theta=1e-10):
    state_space_n = len(env.state_space())
    prev_V = np.zeros(state_space_n, dtype=np.float64)
    
    i = 0 
    while True:
        V = np.zeros(state_space_n, dtype=np.float64)       
        for state in range(state_space_n):
            for action, prob_a in enumerate(pi[state]):
                for prob_s, next_state, reward, done in env.P[state][action]:
                    V[state] += prob_a * prob_s * (reward + gamma * prev_V[next_state] * (not done))

        if np.max(np.abs(V - prev_V)) < theta:
            break
        
        prev_V = V.copy()
        i += 1  
    
    return (V, i)


# Policy iteration: Sutton & Barto (2018), p. 80
def policy_improvement(V, env, gamma=1.0):
    state_space_n = len(env.state_space())
    action_space_n = len(env.action_space())
    Q = np.zeros((state_space_n, action_space_n), dtype=np.float64)
    pi = np.zeros((state_space_n, action_space_n), dtype=np.int32)
    
    # Calculate action-value function for all states 
    for state in range(state_space_n):
        for action in range(action_space_n):
            for prob, next_state, reward, done in env.P[state][action]:
                Q[state][action] += prob * (reward + gamma * V[next_state] * (not done))
    
    # Update policy
    for s in range(len(Q)):
        best_action = np.argmax(Q[s])
        pi[s][best_action] = 1
        
    return pi          

def policy_iteration(env, gamma=1.0, theta=1e-10):
    state_space_n = len(env.state_space())
    action_space_n = len(env.action_space())
    pi = np.ones((state_space_n, action_space_n), dtype=np.float64) / action_space_n
    
    while True:
        old_pi = pi
        V = policy_evaluation(pi, env, gamma, theta)[0]
        pi = policy_improvement(V, env, gamma)
        
        if (old_pi == pi).all():
            break
        
    return pi, V

# Value iteration
def value_iteration(env, gamma=1.0, theta=1e-10):
    return pi, V