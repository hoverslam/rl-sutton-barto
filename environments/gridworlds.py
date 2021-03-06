# Transition function P for some gridworld MDPs
# The outer dictionary keys are the states and the inner ones are the actions with a list of all 
# possible transitions stored in tuples (probability of transition, next state, reward, terminal flag).

import numpy as np

class Grid:
    def __init__(self):
        self.P = None
        self.size = None
        self.max_steps = 100
        self._special_states = {"S": [0], "G": [], "H": [], "W": []}    # (Start, Goals, Holes, Walls) 
        self._initial_state = self._special_states["S"][0]               
        
    def step(self, action):
        if self._steps < self.max_steps:
            prob = []
            for p, _, _, _ in self.P[self._state][action]:
                prob.append(p)
            transition = np.random.choice(len(self.P[self._state][action]), p=prob)    # choose a transition according to probabilities           
            _, next_state, reward, done = self.P[self._state][action][transition]
            self._state = next_state
            self._steps += 1
            return (next_state, reward, done)
        else:
            self._state = self._initial_state
            self._steps = 0
            return (self._initial_state, 0.0, True)

    def reset(self):
        self._initial_state = self._special_states["S"][0]
        self._state = self._initial_state
        self._steps = 0
        return (self._initial_state, 0.0, False)

    def action_space(self):
        return tuple(self.P[0].keys())
    
    def state_space(self):
        return tuple(self.P.keys())
    
    def plot(self, values=None):
        pass
    
class Grid_4x4_Sutton(Grid):
    """ Gridworld 4x4 Sutton: Example 4.1 from Sutton & Barto (2018)
    States: 14 non-terminal, 2 terminal
    Actions: up (0), right (1), down (2), left (3) 
    Reward: -1 for every move
    Actions that would take the agent off the grid do not change the state.
    """
    
    def __init__(self):
        super().__init__()
        self.size = (4, 4)
        self._special_states = {"S": [0], "G": [0, 15], "H": [], "W": []}
        self.P = {
            0: {
                0: [(1.0, 0, 0.0, True)],
                1: [(1.0, 0, 0.0, True)],
                2: [(1.0, 0, 0.0, True)],
                3: [(1.0, 0, 0.0, True)]
            },
            1: {
                0: [(1.0, 1, -1.0, False)],
                1: [(1.0, 2, -1.0, False)],
                2: [(1.0, 5, -1.0, False)],
                3: [(1.0, 0, -1.0, True)]
            },
            2: {
                0: [(1.0, 2, -1.0, False)],
                1: [(1.0, 3, -1.0, False)],
                2: [(1.0, 6, -1.0, False)],
                3: [(1.0, 1, -1.0, False)]
            },
            3: {
                0: [(1.0, 3, -1.0, False)],
                1: [(1.0, 3, -1.0, False)],
                2: [(1.0, 7, -1.0, False)],
                3: [(1.0, 2, -1.0, False)]
            },
            4: {
                0: [(1.0, 0, -1.0, True)],
                1: [(1.0, 5, -1.0, False)],
                2: [(1.0, 8, -1.0, False)],
                3: [(1.0, 4, -1.0, False)]
            },
            5: {
                0: [(1.0, 1, -1.0, False)],
                1: [(1.0, 6, -1.0, False)],
                2: [(1.0, 9, -1.0, False)],
                3: [(1.0, 4, -1.0, False)]
            },
            6: {
                0: [(1.0, 2, -1.0, False)],
                1: [(1.0, 7, -1.0, False)],
                2: [(1.0, 10, -1.0, False)],
                3: [(1.0, 5, -1.0, False)]
            },
            7: {
                0: [(1.0, 3, -1.0, False)],
                1: [(1.0, 7, -1.0, False)],
                2: [(1.0, 11, -1.0, False)],
                3: [(1.0, 6, -1.0, False)]
            },
            8: {
                0: [(1.0, 4, -1.0, False)],
                1: [(1.0, 9, -1.0, False)],
                2: [(1.0, 12, -1.0, False)],
                3: [(1.0, 8, -1.0, False)]
            },
            9: {
                0: [(1.0, 5, -1.0, False)],
                1: [(1.0, 10, -1.0, False)],
                2: [(1.0, 13, -1.0, False)],
                3: [(1.0, 8, -1.0, False)]
            },
            10: {
                0: [(1.0, 6, -1.0, False)],
                1: [(1.0, 11, -1.0, False)],
                2: [(1.0, 14, -1.0, False)],
                3: [(1.0, 9, -1.0, False)]
            },
            11: {
                0: [(1.0, 7, -1.0, False)],
                1: [(1.0, 11, -1.0, False)],
                2: [(1.0, 15, -1.0, True)],
                3: [(1.0, 10, -1.0, False)]
            },
            12: {
                0: [(1.0, 8, -1.0, False)],
                1: [(1.0, 13, -1.0, False)],
                2: [(1.0, 12, -1.0, False)],
                3: [(1.0, 12, -1.0, False)]
            },
            13: {
                0: [(1.0, 9, -1.0, False)],
                1: [(1.0, 14, -1.0, False)],
                2: [(1.0, 13, -1.0, False)],
                3: [(1.0, 12, -1.0, False)]
            },
            14: {
                0: [(1.0, 10, -1.0, False)],
                1: [(1.0, 15, -1.0, True)],
                2: [(1.0, 14, -1.0, False)],
                3: [(1.0, 13, -1.0, False)]
            },
            15: {
                0: [(1.0, 15, 0.0, True)],
                1: [(1.0, 15, 0.0, True)],
                2: [(1.0, 15, 0.0, True)],
                3: [(1.0, 15, 0.0, True)]
            },            
        }
        
class Grid_5x5_Sutton(Grid):
    """ Gridworld 5x5 Sutton: Example 3.5 from Sutton & Barto (2018) 
    States: 25 non-terminal, 0 terminal
    Actions: up (0), right (1), down (2), left (3) 
    Reward: +10 for state 1, +5 for state 3, -1 for stepping out of the grid
    Actions that would take the agent off the grid do not change the state.
    """
    
    def __init__(self):
        super().__init__()
        self.size = (5, 5)
        self.P = {
            0: {
                0: [(1.0, 0, -1.0, False)],
                1: [(1.0, 1, 0.0, False)],
                2: [(1.0, 5, 0.0, False)],
                3: [(1.0, 0, -1.0, False)]
            },
            1: {
                0: [(1.0, 21, 10.0, False)],
                1: [(1.0, 21, 10.0, False)],
                2: [(1.0, 21, 10.0, False)],
                3: [(1.0, 21, 10.0, False)]
            },
            2: {
                0: [(1.0, 2, -1.0, False)],
                1: [(1.0, 3, 0.0, False)],
                2: [(1.0, 7, 0.0, False)],
                3: [(1.0, 1, 0.0, False)]
            },
            3: {
                0: [(1.0, 13, 5.0, False)],
                1: [(1.0, 13, 5.0, False)],
                2: [(1.0, 13, 5.0, False)],
                3: [(1.0, 13, 5.0, False)]
            },
            4: {
                0: [(1.0, 4, -1.0, False)],
                1: [(1.0, 4, -1.0, False)],
                2: [(1.0, 9, 0.0, False)],
                3: [(1.0, 3, 0.0, False)]
            },
            5: {
                0: [(1.0, 0, 0.0, False)],
                1: [(1.0, 6, 0.0, False)],
                2: [(1.0, 10, 0.0, False)],
                3: [(1.0, 5, -1.0, False)]
            },
            6: {
                0: [(1.0, 1, 0.0, False)],
                1: [(1.0, 7, 0.0, False)],
                2: [(1.0, 11, 0.0, False)],
                3: [(1.0, 5, 0.0, False)]
            },
            7: {
                0: [(1.0, 2, 0.0, False)],
                1: [(1.0, 8, 0.0, False)],
                2: [(1.0, 12, 0.0, False)],
                3: [(1.0, 6, 0.0, False)]
            },
            8: {
                0: [(1.0, 3, 0.0, False)],
                1: [(1.0, 9, 0.0, False)],
                2: [(1.0, 13, 0.0, False)],
                3: [(1.0, 7, 0.0, False)]
            },
            9: {
                0: [(1.0, 4, 0.0, False)],
                1: [(1.0, 9, -1.0, False)],
                2: [(1.0, 14, 0.0, False)],
                3: [(1.0, 8, 0.0, False)]
            },
            10: {
                0: [(1.0, 5, 0.0, False)],
                1: [(1.0, 11, 0.0, False)],
                2: [(1.0, 15, 0.0, False)],
                3: [(1.0, 10, -1.0, False)]
            },
            11: {
                0: [(1.0, 6, 0.0, False)],
                1: [(1.0, 12, 0.0, False)],
                2: [(1.0, 16, 0.0, False)],
                3: [(1.0, 10, 0.0, False)]
            },
            12: {
                0: [(1.0, 7, 0.0, False)],
                1: [(1.0, 13, 0.0, False)],
                2: [(1.0, 17, 0.0, False)],
                3: [(1.0, 11, 0.0, False)]
            },
            13: {
                0: [(1.0, 8, 0.0, False)],
                1: [(1.0, 14, 0.0, False)],
                2: [(1.0, 18, 0.0, False)],
                3: [(1.0, 12, 0.0, False)]
            },
            14: {
                0: [(1.0, 9, 0.0, False)],
                1: [(1.0, 14, -1.0, False)],
                2: [(1.0, 19, 0.0, False)],
                3: [(1.0, 13, 0.0, False)]
            },
            15: {
                0: [(1.0, 10, 0.0, False)],
                1: [(1.0, 16, 0.0, False)],
                2: [(1.0, 20, 0.0, False)],
                3: [(1.0, 15, -1.0, False)]
            },
            16: {
                0: [(1.0, 11, 0.0, False)],
                1: [(1.0, 17, 0.0, False)],
                2: [(1.0, 21, 0.0, False)],
                3: [(1.0, 15, 0.0, False)]
            },
            17: {
                0: [(1.0, 12, 0.0, False)],
                1: [(1.0, 18, 0.0, False)],
                2: [(1.0, 22, 0.0, False)],
                3: [(1.0, 16, 0.0, False)]
            },
            18: {
                0: [(1.0, 13, 0.0, False)],
                1: [(1.0, 19, 0.0, False)],
                2: [(1.0, 23, 0.0, False)],
                3: [(1.0, 17, 0.0, False)]
            },
            19: {
                0: [(1.0, 14, 0.0, False)],
                1: [(1.0, 19, -1.0, False)],
                2: [(1.0, 24, 0.0, False)],
                3: [(1.0, 18, 0.0, False)]
            },
            20: {
                0: [(1.0, 15, 0.0, False)],
                1: [(1.0, 21, 0.0, False)],
                2: [(1.0, 20, -1.0, False)],
                3: [(1.0, 20, -1.0, False)]
            },
            21: {
                0: [(1.0, 16, 0.0, False)],
                1: [(1.0, 22, 0.0, False)],
                2: [(1.0, 21, -1.0, False)],
                3: [(1.0, 20, 0.0, False)]
            },
            22: {
                0: [(1.0, 17, 0.0, False)],
                1: [(1.0, 23, 0.0, False)],
                2: [(1.0, 22, -1.0, False)],
                3: [(1.0, 21, 0.0, False)]
            },
            23: {
                0: [(1.0, 18, 0.0, False)],
                1: [(1.0, 24, 0.0, False)],
                2: [(1.0, 23, -1.0, False)],
                3: [(1.0, 22, 0.0, False)]
            },
            24: {
                0: [(1.0, 19, 0.0, False)],
                1: [(1.0, 24, -1.0, False)],
                2: [(1.0, 24, -1.0, False)],
                3: [(1.0, 23, 0.0, False)]
            },
        }      
       
class Grid_3x4_RNG(Grid):
    """ Gridworld 3x4 RNG: Russell, Norvig (2020) Artificial Intelligence: A Modern Approach, see Figure 23.1
    States: 9 non-terminal, 2 terminal, 1 wall
    Actions: up (0), right (1), down (2), left (3) 
    Reward: -0.04 for every move, +1 for reaching goal, -1 for move in hole
    Actions that would take the agent off the grid do not change the state.
    """
    
    def __init__(self):
        super().__init__()
        self.size = (3, 4)
        self._special_states = {"S": [0], "G": [3], "H": [7], "W": [5]}
        self.P = {
            0: {
                0: [(0.9, 0, -0.04, False), (0.1, 1, -0.04, False)],
                1: [(0.8, 1, -0.04, False), (0.1, 4, -0.04, False), (0.1, 0, -0.04, False)],
                2: [(0.8, 4, -0.04, False), (0.1, 1, -0.04, False), (0.1, 0, -0.04, False)],
                3: [(0.9, 0, -0.04, False), (0.1, 4, -0.04, False)]
            },
            1: {
                0: [(0.8, 1, -0.04, False), (0.1, 2, -0.04, False), (0.1, 0, -0.04, False)],
                1: [(0.8, 2, -0.04, False), (0.2, 1, -0.04, False)],
                2: [(0.8, 1, -0.04, False), (0.1, 0, -0.04, False), (0.1, 2, -0.04, False)],
                3: [(0.8, 0, -0.04, False), (0.2, 1, -0.04, False)]
            },
            2: {
                0: [(0.8, 2, -0.04, False), (0.1, 1, -0.04, False), (0.1, 3, 1.0, True)],
                1: [(0.8, 3, 1.0, True), (0.1, 2, -0.04, False), (0.1, 6, -0.04, False)],
                2: [(0.8, 6, -0.04, False), (0.1, 1, -0.04, False), (0.1, 3, 1.0, True)],
                3: [(0.8, 1, -0.04, False), (0.1, 2, -0.04, False), (0.1, 6, -0.04, False)]
            },
            3: {
                0: [(1.0, 3, 0.0, True)],
                1: [(1.0, 3, 0.0, True)],
                2: [(1.0, 3, 0.0, True)],
                3: [(1.0, 3, 0.0, True)]
            },
            4: {
                0: [(0.8, 0, -0.04, False), (0.2, 4, -0.04, False)],
                1: [(0.8, 4, -0.04, False), (0.1, 0, -0.04, False), (0.1, 8, -0.04, False)],
                2: [(0.8, 8, -0.04, False), (0.2, 4, -0.04, False)],
                3: [(0.8, 4, -0.04, False), (0.1, 0, -0.04, False), (0.1, 8, -0.04, False)]
            },
            5: {
                0: [(1.0, 5, 0.0, True)],
                1: [(1.0, 5, 0.0, True)],
                2: [(1.0, 5, 0.0, True)],
                3: [(1.0, 5, 0.0, True)]
            },
            6: {
                0: [(0.8, 2, -0.04, False), (0.1, 7, -1.0, True), (0.1, 6, -0.04, False)],
                1: [(0.8, 7, -1.0, True), (0.1, 2, -0.04, False), (0.1, 10, -0.04, False)],
                2: [(0.8, 10, -0.04, False), (0.1, 7, -1.0, True), (0.1, 6, -0.04, False)],
                3: [(0.8, 6, -0.04, False), (0.1, 2, -0.04, False), (0.1, 10, -0.04, False)]
            },
            7: {
                0: [(1.0, 7, 0.0, True)],
                1: [(1.0, 7, 0.0, True)],
                2: [(1.0, 7, 0.0, True)],
                3: [(1.0, 7, 0.0, True)]
            },
            8: {
                0: [(0.8, 4, -0.04, False), (0.1, 9, -0.04, False), (0.1, 8, -0.04, False)],
                1: [(0.8, 9, -0.04, False), (0.1, 8, -0.04, False), (0.1, 4, -0.04, False)],
                2: [(0.9, 8, -0.04, False), (0.1, 9, -0.04, False)],
                3: [(0.9, 8, -0.04, False), (0.1, 4, -0.04, False)]
            },
            9: {
                0: [(0.8, 9, -0.04, False), (0.1, 10, -0.04, False), (0.1, 8, -0.04, False)],
                1: [(0.8, 10, -0.04, False), (0.2, 9, -0.04, False)],
                2: [(0.8, 9, -0.04, False), (0.1, 10, -0.04, False), (0.1, 8, -0.04, False)],
                3: [(0.8, 8, -0.04, False), (0.2, 9, -0.04, False)]
            },
            10: {
                0: [(0.8, 6, -0.04, False), (0.1, 11, -0.04, False), (0.1, 9, -0.04, False)],
                1: [(0.8, 11, -0.04, False), (0.1, 6, -0.04, False), (0.1, 10, -0.04, False)],
                2: [(0.8, 10, -0.04, False), (0.1, 9, -0.04, False), (0.1, 11, -0.04, False)],
                3: [(0.8, 9, -0.04, False), (0.1, 6, -0.04, False), (0.1, 10, -0.04, False)]
            },
            11: {
                0: [(0.8, 7, -1.0, True), (0.1, 10, -0.04, False), (0.1, 11, -0.04, False)],
                1: [(0.9, 11, -0.04, False), (0.1, 7, -1.0, True)],
                2: [(0.9, 11, -0.04, False), (0.1, 10, -0.04, False)],
                3: [(0.8, 10, -0.04, False), (0.1, 7, -1.0, True), (0.1, 11, -0.04, False)]
            }
        }
        
class Cliff_Walking(Grid):
    """ Cliff Walking: Example 6.6 from Sutton & Barto (2018) 
    States: 47 non-terminal, 1 terminal
    Actions: up (0), right (1), down (2), left (3) 
    Reward: -1 for every move, -100 falling down the cliff
    Actions that would take the agent off the grid do not change the state.
    """
        
    def __init__(self):
        super().__init__()
        self.size = (4, 12)
        self._special_states = {"S": [36], "G": [47], "H": [37, 38, 39, 40, 41, 42, 43, 44, 45, 46], "W": []}
        self.P = {
            0: {
                0: [(1.0, 0, -1.0, False)],
                1: [(1.0, 1, -1.0, False)],
                2: [(1.0, 12, -1.0, False)],
                3: [(1.0, 0, -1.0, False)]
            },
            1: {
                0: [(1.0, 1, -1.0, False)],
                1: [(1.0, 2, -1.0, False)],
                2: [(1.0, 13, -1.0, False)],
                3: [(1.0, 0, -1.0, False)]
            },
            2: {
                0: [(1.0, 2, -1.0, False)],
                1: [(1.0, 3, -1.0, False)],
                2: [(1.0, 14, -1.0, False)],
                3: [(1.0, 1, -1.0, False)]
            },
            3: {
                0: [(1.0, 3, -1.0, False)],
                1: [(1.0, 4, -1.0, False)],
                2: [(1.0, 15, -1.0, False)],
                3: [(1.0, 2, -1.0, False)]
            },
            4: {
                0: [(1.0, 4, -1.0, False)],
                1: [(1.0, 5, -1.0, False)],
                2: [(1.0, 16, -1.0, False)],
                3: [(1.0, 3, -1.0, False)]
            },
            5: {
                0: [(1.0, 5, -1.0, False)],
                1: [(1.0, 6, -1.0, False)],
                2: [(1.0, 17, -1.0, False)],
                3: [(1.0, 4, -1.0, False)]
            },
            6: {
                0: [(1.0, 6, -1.0, False)],
                1: [(1.0, 7, -1.0, False)],
                2: [(1.0, 18, -1.0, False)],
                3: [(1.0, 5, -1.0, False)]
            },
            7: {
                0: [(1.0, 7, -1.0, False)],
                1: [(1.0, 8, -1.0, False)],
                2: [(1.0, 19, -1.0, False)],
                3: [(1.0, 6, -1.0, False)]
            },
            8: {
                0: [(1.0, 8, -1.0, False)],
                1: [(1.0, 9, -1.0, False)],
                2: [(1.0, 20, -1.0, False)],
                3: [(1.0, 7, -1.0, False)]
            },
            9: {
                0: [(1.0, 9, -1.0, False)],
                1: [(1.0, 10, -1.0, False)],
                2: [(1.0, 21, -1.0, False)],
                3: [(1.0, 8, -1.0, False)]
            },
            10: {
                0: [(1.0, 10, -1.0, False)],
                1: [(1.0, 11, -1.0, False)],
                2: [(1.0, 22, -1.0, False)],
                3: [(1.0, 9, -1.0, False)]
            },
            11: {
                0: [(1.0, 11, -1.0, False)],
                1: [(1.0, 11, -1.0, False)],
                2: [(1.0, 23, -1.0, False)],
                3: [(1.0, 10, -1.0, False)]
            },
            12: {
                0: [(1.0, 0, -1.0, False)],
                1: [(1.0, 13, -1.0, False)],
                2: [(1.0, 24, -1.0, False)],
                3: [(1.0, 12, -1.0, False)]
            },
            13: {
                0: [(1.0, 1, -1.0, False)],
                1: [(1.0, 14, -1.0, False)],
                2: [(1.0, 25, -1.0, False)],
                3: [(1.0, 12, -1.0, False)]
            },
            14: {
                0: [(1.0, 2, -1.0, False)],
                1: [(1.0, 15, -1.0, False)],
                2: [(1.0, 26, -1.0, False)],
                3: [(1.0, 13, -1.0, False)]
            },
            15: {
                0: [(1.0, 3, -1.0, False)],
                1: [(1.0, 16, -1.0, False)],
                2: [(1.0, 27, -1.0, False)],
                3: [(1.0, 14, -1.0, False)]
            },
            16: {
                0: [(1.0, 4, -1.0, False)],
                1: [(1.0, 17, -1.0, False)],
                2: [(1.0, 28, -1.0, False)],
                3: [(1.0, 15, -1.0, False)]
            },
            17: {
                0: [(1.0, 5, -1.0, False)],
                1: [(1.0, 18, -1.0, False)],
                2: [(1.0, 29, -1.0, False)],
                3: [(1.0, 16, -1.0, False)]
            },
            18: {
                0: [(1.0, 6, -1.0, False)],
                1: [(1.0, 19, -1.0, False)],
                2: [(1.0, 30, -1.0, False)],
                3: [(1.0, 17, -1.0, False)]
            },
            19: {
                0: [(1.0, 7, -1.0, False)],
                1: [(1.0, 20, -1.0, False)],
                2: [(1.0, 31, -1.0, False)],
                3: [(1.0, 18, -1.0, False)]
            },
            20: {
                0: [(1.0, 8, -1.0, False)],
                1: [(1.0, 21, -1.0, False)],
                2: [(1.0, 32, -1.0, False)],
                3: [(1.0, 19, -1.0, False)]
            },
            21: {
                0: [(1.0, 9, -1.0, False)],
                1: [(1.0, 22, -1.0, False)],
                2: [(1.0, 33, -1.0, False)],
                3: [(1.0, 20, -1.0, False)]
            },
            22: {
                0: [(1.0, 10, -1.0, False)],
                1: [(1.0, 23, -1.0, False)],
                2: [(1.0, 34, -1.0, False)],
                3: [(1.0, 21, -1.0, False)]
            },
            23: {
                0: [(1.0, 11, -1.0, False)],
                1: [(1.0, 23, -1.0, False)],
                2: [(1.0, 35, -1.0, False)],
                3: [(1.0, 22, -1.0, False)]
            },
            24: {
                0: [(1.0, 12, -1.0, False)],
                1: [(1.0, 25, -1.0, False)],
                2: [(1.0, 36, -1.0, False)],
                3: [(1.0, 24, -1.0, False)]
            },
            25: {
                0: [(1.0, 13, -1.0, False)],
                1: [(1.0, 26, -1.0, False)],
                2: [(1.0, 36, -100.0, False)],
                3: [(1.0, 24, -1.0, False)]
            },
            26: {
                0: [(1.0, 14, -1.0, False)],
                1: [(1.0, 27, -1.0, False)],
                2: [(1.0, 36, -100.0, False)],
                3: [(1.0, 25, -1.0, False)]
            },
            27: {
                0: [(1.0, 15, -1.0, False)],
                1: [(1.0, 28, -1.0, False)],
                2: [(1.0, 36, -100.0, False)],
                3: [(1.0, 26, -1.0, False)]
            },
            28: {
                0: [(1.0, 16, -1.0, False)],
                1: [(1.0, 29, -1.0, False)],
                2: [(1.0, 36, -100.0, False)],
                3: [(1.0, 27, -1.0, False)]
            },
            29: {
                0: [(1.0, 17, -1.0, False)],
                1: [(1.0, 30, -1.0, False)],
                2: [(1.0, 36, -100.0, False)],
                3: [(1.0, 28, -1.0, False)]
            },
            30: {
                0: [(1.0, 18, -1.0, False)],
                1: [(1.0, 31, -1.0, False)],
                2: [(1.0, 36, -100.0, False)],
                3: [(1.0, 29, -1.0, False)]
            },
            31: {
                0: [(1.0, 19, -1.0, False)],
                1: [(1.0, 32, -1.0, False)],
                2: [(1.0, 36, -100.0, False)],
                3: [(1.0, 30, -1.0, False)]
            },
            32: {
                0: [(1.0, 20, -1.0, False)],
                1: [(1.0, 33, -1.0, False)],
                2: [(1.0, 36, -100.0, False)],
                3: [(1.0, 31, -1.0, False)]
            },
            33: {
                0: [(1.0, 21, -1.0, False)],
                1: [(1.0, 34, -1.0, False)],
                2: [(1.0, 36, -100.0, False)],
                3: [(1.0, 32, -1.0, False)]
            },
            34: {
                0: [(1.0, 22, -1.0, False)],
                1: [(1.0, 35, -1.0, False)],
                2: [(1.0, 36, -100.0, False)],
                3: [(1.0, 33, -1.0, False)]
            },
            35: {
                0: [(1.0, 23, -1.0, False)],
                1: [(1.0, 35, -1.0, False)],
                2: [(1.0, 47, -1.0, True)],
                3: [(1.0, 34, -1.0, False)]
            },
            36: {
                0: [(1.0, 24, -1.0, False)],
                1: [(1.0, 36, -100.0, False)],
                2: [(1.0, 36, -1.0, False)],
                3: [(1.0, 36, -1.0, False)]
            },
            37: {
                0: [(1.0, 37, 0.0, True)],
                1: [(1.0, 37, 0.0, True)],
                2: [(1.0, 37, 0.0, True)],
                3: [(1.0, 37, 0.0, True)]
            },
            38: {
                0: [(1.0, 38, 0.0, True)],
                1: [(1.0, 38, 0.0, True)],
                2: [(1.0, 38, 0.0, True)],
                3: [(1.0, 38, 0.0, True)]
            },
            39: {
                0: [(1.0, 39, 0.0, True)],
                1: [(1.0, 39, 0.0, True)],
                2: [(1.0, 39, 0.0, True)],
                3: [(1.0, 39, 0.0, True)]
            },
            40: {
                0: [(1.0, 40, 0.0, True)],
                1: [(1.0, 40, 0.0, True)],
                2: [(1.0, 40, 0.0, True)],
                3: [(1.0, 40, 0.0, True)]
            },
            41: {
                0: [(1.0, 41, 0.0, True)],
                1: [(1.0, 41, 0.0, True)],
                2: [(1.0, 41, 0.0, True)],
                3: [(1.0, 41, 0.0, True)]
            },
            42: {
                0: [(1.0, 42, 0.0, True)],
                1: [(1.0, 42, 0.0, True)],
                2: [(1.0, 42, 0.0, True)],
                3: [(1.0, 42, 0.0, True)]
            },
            43: {
                0: [(1.0, 43, 0.0, True)],
                1: [(1.0, 43, 0.0, True)],
                2: [(1.0, 43, 0.0, True)],
                3: [(1.0, 43, 0.0, True)]
            },
            44: {
                0: [(1.0, 44, 0.0, True)],
                1: [(1.0, 44, 0.0, True)],
                2: [(1.0, 44, 0.0, True)],
                3: [(1.0, 44, 0.0, True)]
            },
            45: {
                0: [(1.0, 45, 0.0, True)],
                1: [(1.0, 45, 0.0, True)],
                2: [(1.0, 45, 0.0, True)],
                3: [(1.0, 45, 0.0, True)]
            },
            46: {
                0: [(1.0, 46, 0.0, True)],
                1: [(1.0, 46, 0.0, True)],
                2: [(1.0, 46, 0.0, True)],
                3: [(1.0, 46, 0.0, True)]
            },
            47: {
                0: [(1.0, 47, 0.0, True)],
                1: [(1.0, 47, 0.0, True)],
                2: [(1.0, 47, 0.0, True)],
                3: [(1.0, 47, 0.0, True)]
            }
        }
        
        
