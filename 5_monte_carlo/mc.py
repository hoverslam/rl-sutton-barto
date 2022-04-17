import numpy as np
from tqdm import tqdm

def generate_trajectory(pi, env):
    trajectoy = []
    state, reward, done = env.reset()
    while True:
        action = pi(state)
        next_state, reward, done = env.step(action)
        if not done:
            trajectoy.append((state, action, next_state, reward, done))
            state = next_state
        else:
            break
        
    return np.array(trajectoy, dtype=object)     

# Monte Carlo Prediction: Sutton & Barto (2018), page 92
def mc_prediction(pi, env, gamma=1.0, episodes=500, first_visit=True):
    nS = len(env.state_space())
    V = np.zeros(nS, dtype=np.float64)
    returns = np.zeros(nS, dtype=np.float64)
    counts = np.zeros(nS, dtype=np.float64)
    
    for e in tqdm(range(episodes)):
        trajectory = generate_trajectory(pi, env)
        visited = np.zeros(nS, dtype=bool)
        G = 0        
        for state, _, _, reward, _ in trajectory:
            G = gamma * G + reward
            if visited[state] and first_visit:
                continue            
            visited[state] = True
            returns[state] += G
            counts[state] += 1
    
    for i in range(len(V)):
        V[i] = returns[i] / counts[i]
    
    return V