import numpy as np
from tqdm import tqdm


# Monte Carlo Prediction: Sutton & Barto (2018), page 92
    # [TODO]


# GLIE Monte Carlo Control: Introduction to RL with David Silver (2015), Lecture 5
def epsilon_greedy(Q, state, epsilon):
    if np.random.rand() < epsilon:
        return np.random.randint(Q.shape[1])
    else:
        return np.argmax(Q[state])
    
def generate_trajectory(env, Q, epsilon):
    trajectoy = []
    state, reward, done = env.reset()
    while True:
        action = epsilon_greedy(Q, state, epsilon)
        next_state, reward, done = env.step(action)
        if not done:
            trajectoy.append((state, action, next_state, reward, done))
            state = next_state
        else:
            break
        
    return np.array(trajectoy, dtype=object)  

def mc_control(env, gamma=1.0, episodes=500):
    nS = len(env.state_space())
    nA = len(env.action_space())
    Q = np.zeros((nS, nA), dtype=np.float64)
    counts = np.zeros((nS, nA), dtype=np.int32)
    discounts = np.logspace(0, env.max_steps, num=env.max_steps, base=gamma, endpoint=False)
    
    for e in tqdm(range(episodes)):      
        epsilon = 1 / (e + 1)
        trajectory = generate_trajectory(env, Q, epsilon)
        for t, (state, action, _, reward, _) in enumerate(trajectory):            
            counts[state][action] += 1
            steps = len(trajectory[t:])
            G = np.sum(discounts[:steps] * trajectory[t:, 3])
            Q[state][action] = Q[state][action] + (1 / counts[state][action]) * (G - Q[state][action])
            
    pi = np.argmax(Q, axis=1)

    return (pi, Q)