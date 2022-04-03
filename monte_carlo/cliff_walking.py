# The "Cliff Walking" gridworld from Sutton & Barto (2020), example 6.6

class TheCliff:
    def __init__(self):
        self.height = 4
        self.width = 12
        self.actions = {
            0: (-1, 0),     # up
            1: (0, 1),      # right
            2: (1, 0),      # down
            3: (0, -1)      # left
        }
        self.reset()
    
    def step(self, action):
        x, y = self.actions[action]
        self.S = (self.S[0] + x, self.S[1] + y)
        
        # Make sure to not step out of the grid
        self.S = (max(self.S[0], 0), max(self.S[1], 0))
        self.S = (min(self.S[0], self.height - 1), min(self.S[1], self.width - 1))
        
        # Return next state, reward and done flag for a given action
        if self.S == (self.height - 1, self.width - 1):
            return (self.S, -1.0, True)             # goal
        elif self.S[1] != 0 and self.S[0] == self.height - 1:
            return (self.reset(), -100.0, False)    # cliff
        else:
            return (self.S, -1.0, False)
        
    def action_space(self):
        return self.actions.keys()
    
    def state_space(self):
        states = []
        for x in range(self.height):
            for y in range(self.width):
                states.append((x, y))
        return states
    
    def reset(self):
        self.S = (3, 0)
        return self.S