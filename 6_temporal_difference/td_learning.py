import numpy as np
from tqdm import tqdm


# Policy
def epsilon_greedy(Q, state, epsilon):
    if np.random.rand() < epsilon:
        return np.random.randint(Q.shape[1])
    else:
        return np.argmax(Q[state])


# SARSA: Sutton & Barto (2018), p. 130
def sarsa(env, gamma=1.0, epsilon= 0.1, alpha=0.1, episodes=10000):
    nS = len(env.state_space())
    nA = len(env.action_space())
    Q = np.zeros((nS, nA), dtype=np.float64)
    
    for e in tqdm(range(episodes)):
        state, reward, done = env.reset()
        action = epsilon_greedy(Q, state, epsilon)        
                
        while not done:
            next_state, reward, done = env.step(action)
            next_action = epsilon_greedy(Q, state, epsilon)
            
            # Update Q values using the action taken in the next state
            td_target = reward + gamma * Q[next_state][next_action] * (not done)
            td_error = td_target - Q[state][action]
            Q[state][action] += alpha * td_error

            state, action = next_state, next_action
            
    pi = np.argmax(Q, axis=1)
            
    return (pi, Q)


# SARSA with GLIE: Introduction to RL with David Silver (2015), Lecture 5
def sarsa_glie(env, gamma=1.0, alpha=0.1, episodes=10000):
    nS = len(env.state_space())
    nA = len(env.action_space())
    Q = np.zeros((nS, nA), dtype=np.float64)
    
    for e in tqdm(range(episodes)):
        epsilon = 1 / (e + 1)    # GLIE -> Greedy in the Limit with Infinite Exploration
        state, reward, done = env.reset()
        action = epsilon_greedy(Q, state, epsilon)        
                
        while not done:
            next_state, reward, done = env.step(action)
            next_action = epsilon_greedy(Q, state, epsilon)
            
            # Update Q values using the action taken in the next state
            td_target = reward + gamma * Q[next_state][next_action] * (not done)
            td_error = td_target - Q[state][action]
            Q[state][action] += alpha * td_error

            state, action = next_state, next_action
            
    pi = np.argmax(Q, axis=1)
            
    return (pi, Q)


# Expected SARSA: Sutton & Barto (2018), p. 133
def expected_sarsa(env, gamma=1.0, epsilon=0.1, alpha=0.1, episodes=10000):
    nS = len(env.state_space())
    nA = len(env.action_space())
    Q = np.zeros((nS, nA), dtype=np.float64)
    
    for e in tqdm(range(episodes)):
        state, reward, done = env.reset()               
                
        while not done:
            action = epsilon_greedy(Q, state, epsilon) 
            next_state, reward, done = env.step(action)
            
            # Probability distribution for next action
            prob = np.full(nA, epsilon / nA, dtype=np.float64)
            prob[np.argmax(Q[next_state])] += 1 - epsilon
            
            # Update Q values using the expected value of the next state
            td_target = reward + gamma * np.sum(prob * Q[next_state]) * (not done)
            td_error = td_target - Q[state][action]
            Q[state][action] += alpha * td_error

            state = next_state
            
    pi = np.argmax(Q, axis=1)
            
    return (pi, Q)


# Q-Learning: Sutton & Barto (2018), p. 131
def q_learning(env, gamma=1.0, epsilon = 0.1, alpha=0.1, episodes=10000):
    nS = len(env.state_space())
    nA = len(env.action_space())
    Q = np.zeros((nS, nA), dtype=np.float64)
    
    for e in tqdm(range(episodes)):        
        state, reward, done = env.reset()
                
        while not done:
            action = epsilon_greedy(Q, state, epsilon)  
            next_state, reward, done = env.step(action)
            
            # Update Q values using the action with the highest value in the next state
            td_target = reward + gamma * np.max(Q[next_state]) * (not done)
            td_error = td_target - Q[state][action]
            Q[state][action] += alpha * td_error

            state = next_state
            
    pi = np.argmax(Q, axis=1)
            
    return (pi, Q)


# Double Q-learning: Sutton & Barto (2018), p. 136
def double_q_learning(env, gamma=1.0, epsilon = 0.1, alpha=0.1, episodes=10000):
    nS = len(env.state_space())
    nA = len(env.action_space())
    Q1 = np.zeros((nS, nA), dtype=np.float64)
    Q2 = np.zeros((nS, nA), dtype=np.float64)
    
    for e in tqdm(range(episodes)):        
        state, reward, done = env.reset()              
                
        while not done:
            action = epsilon_greedy(np.add(Q1, Q2), state, epsilon) 
            next_state, reward, done = env.step(action)
            
            # Update Q1 values using Q2 or vice versa with a probability of 0.5
            if np.random.rand() < 0.5:
                td_target = reward + gamma * Q2[next_state][np.argmax(Q1[next_state])] * (not done)
                td_error = td_target - Q1[state][action]
                Q1[state][action] += alpha * td_error
            else:                
                td_target = reward + gamma * Q1[next_state][np.argmax(Q2[next_state])] * (not done)
                td_error = td_target - Q2[state][action]
                Q2[state][action] += alpha * td_error

            state = next_state
    
    Q = np.add(Q1, Q2) / 2   # Take the average of Q1 and Q2 as the final Q values        
    pi = np.argmax(Q, axis=1)
            
    return (pi, Q)


# Speedy Q-learning: Ghavamzadeh, et. al (2011) Speedy Q-learning
# [TODO]