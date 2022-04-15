import numpy as np

# Policy evaluation: Sutton & Barto (2018), p. 75
def policy_evaluation(pi, env, gamma=1, theta=1e-10):
    state_space_n = len(env.state_space())
    prev_V = np.zeros(state_space_n, dtype=np.float32)
    
    i = 0 
    while True:
        V = np.zeros(state_space_n, dtype=np.float32)       
        for state in range(state_space_n):
            for action, prob_a in enumerate(pi[state]):
                for prob_s, next_state, reward, done in env.P[state][action]:
                    V[state] += prob_a * prob_s * (reward + gamma * prev_V[next_state] * (not done))

        if np.max(np.abs(V - prev_V)) < theta:
            break
        
        prev_V = V.copy()
        i += 1  
    
    return (V, i)


# Policy iteration
def policy_iteration(env, gamma=1, theta=10e-10):
    # [TODO]
    return None


# Value iteration
def value_iteration(env, gamma=1, theta=10e-10):
    # [TODO]
    return None