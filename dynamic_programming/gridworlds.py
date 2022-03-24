# Transition function P for some gridworld MDPs
# The outer dictionary keys are the states and the inner ones are the actions with a list of all 
# possible transitions stored in tuples (probability of transition, next state, reward, terminal flag).

import numpy as np

class Grid:
    def __init__(self):
        self.size = None
        self.P = None
        
    def print_values(self, values, digits=1):
        values = np.reshape(values, self.size)
        values = np.round(values, digits)
        print(values)  

class Grid_4x4(Grid):
    """ Gridworld 4x4:
    States: 14 non-terminal, 2 terminal
    Actions: up (0), right (1), down (2), left (3) 
    Reward: -1 for every move
    Actions that would take the agent off the grid leave its location unchanged.
    """
    
    def __init__(self):
        super().__init__()
        self.size = (4, 4)
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
    """ Gridworld 5x5 Sutton: Example 3.5 from the book 
    States: 25 non-terminal, 0 terminal
    Actions: up (0), right (1), down (2), left (3) 
    Reward: +10 for state 1, +5 for state 3, -1 for stepping out of the grid
    Actions that would take the agent off the grid leave its location unchanged.
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