import numpy as np
import cliff_walking
from tqdm import tqdm

def epsilon_greedy(state, Q, epsilon):
    if np.random.rand() < epsilon:
        return np.random.randint(0, Q.shape[2])
    else:   
        return np.argmax(Q[state[0]][state[1]])

def get_policy(Q):
    return np.argmax(Q, axis=2)

def sarsa(env, gamma=1.0, alpha=0.1, epsilon=0.1, episodes=10000):
    dim = (env.height, env.width, len(env.action_space()))   
    Q = np.zeros(dim, dtype=np.float32)

    for e in tqdm(range(episodes)):
        state = env.reset()
        done = False
        action = epsilon_greedy(state, Q, epsilon)
        
        while not done:
            next_state, reward, done = env.step(action)
            next_action = epsilon_greedy(next_state, Q, epsilon)
            
            # Update Q values using the action taken in the next state
            td_target = reward + gamma * Q[next_state[0]][next_state[1]][next_action] * (not done)
            td_error = td_target - Q[state[0]][state[1]][action]
            Q[state[0]][state[1]][action] += alpha * td_error
            
            state, action = next_state, next_action
            
    return Q


env = cliff_walking.TheCliff()
Q = sarsa(env, episodes=100000)
print(get_policy(Q))