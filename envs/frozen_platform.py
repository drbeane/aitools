import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm


class FrozenPlatform():
    
    def __init__(self, rows, cols, sp_range, start=1, goal=None, holes=0, 
                 random_state=None, is_copy=False):
        '''
        Atributes:
        rows            Number of rows in platform.
        cols            Number of cols in platform.
        sp_range        Range of slip % values for cells.
        start           Index of starting cell.
        goal            Index of goal cell.
        holes           Either an integer number of randomly place holes or
                        a list of cell indices for hole locations.
        random_state    Random seed governing platform generation. 
        is_copy         Used to skip some steps when creating a copy. 
        '''
        
        # Store the platform geometry attributes.
        self.rows = rows
        self.cols = cols
        
        # There are rows*cols cells on the platform
        # There is another state of 0 to indicate being off the platform
        self.num_states = rows * cols + 1
        self.num_actions = 4
        self.sp_range = sp_range    
        self.start = start
        self.goal = rows * cols if goal is None else goal
        self.actions = [0,1,2,3]  # U,R,D,L
        
        # Establish the reward structure
        self.reward = 100
        self.penalty = -100
        self.step_penalty = -1
        
        # Skip the rest of the constructor if this instance is a copy
        if is_copy: return
        
        # Set current state and terminal status. 
        self.state = self.start     
        self.terminal = False
        
        # Set random state for genation of holes and slip probabilities
        if random_state is not None:
            np_state = np.random.get_state()
            np.random.seed(random_state)
        
        # Generate Holes
        
        # The lines below are a dumb hack to cause the seed to behave the same when
        # setting holes=0, holes=None, and holes=[]. 
        if (holes == []) or (holes is None):
            holes = 0            
        
        if type(holes) == int:
            candidates = [
                x for x in range(1, self.num_states) 
                if x not in [self.start, self.goal]
            ]
            self.holes = list(np.random.choice(candidates, holes, replace=False))
        elif type(holes) == list:
            self.holes = holes
        else:
            self.holes = []
        
        # Generate Slip Probabilites
        self.slip_prob = np.random.uniform(
            low=sp_range[0], high = sp_range[1], size=self.num_states
        ).round(2)
        for s in list(self.holes) + [0, self.goal]:
            self.slip_prob[s] = 0
        
        # Generate Dynamics
        self.dynamics = []
        for s in range(self.num_states):   # Loop over states
            sp = self.slip_prob[s]
            q = round(sp / 3, 2) # P[Specific Wrong Dir] 
            p = round(1 - 3*q,2)             # P[Correct Direction]
            results_by_action = [
                {
                    'prob':[(p if e==a else q) for e in range(4)], 
                    'next_state':[self.next_state(s,e) for e in range(4)]
                }
                for a in self.actions
            ]
            self.dynamics.append(results_by_action)
        
        # Reset previous random state, if needed.         
        if random_state is not None:
            np.random.set_state(np_state)
            
        # History objects
        self.actions_taken = []
        self.path = [start]
        self.rewards = []
    
       
    def copy(self, set_start=None):
        new_node = FrozenPlatform(self.rows, self.cols, self.sp_range, 
                                  self.start, self.goal, is_copy=True)
        
        # Copy platform information and dynamics
        new_node.holes = self.holes
        new_node.slip_prob = self.slip_prob
        new_node.dynamics = self.dynamics
        
        
        # Copy current state information
        new_node.state = self.state
        new_node.terminal = self.terminal
                    
        # Copy history information
        new_node.actions_taken = [*self.actions_taken]
        new_node.path = [*self.path]
        new_node.rewards = [*self.rewards]
        
        return new_node

   
    def reset(self, set_start=None):
        '''
        Returns a copy of the instance, but with a new starting point. 
        This is used with MC and TD methods at the start of each new episode. 
        '''
        new_node = self.copy()
        start = self.start if set_start is None else set_start
        
        new_node.state = start
        new_node.actions_taken = []
        new_node.path = [start]
        new_node.rewards = []
        
        return new_node


    def next_state(self, s, e):
        ''' 
        Returns the next state obtained by applying effect e in state s. 
        Note that e is an actual effect, not an action taken. 
        '''
        
        # If effect is being applied from a hole or goal, return state. 
        if s in list(self.holes) + [0, self.goal]:
            return s
        
        # Obtain row number and column for state. 
        row_num = (s-1) // self.cols
        col_num = (s-1) % self.cols
       
        # Check to see if we are moving off the board
        b1 = (e == 0 and row_num == 0) 
        b2 = (e == 2 and row_num == self.rows-1)
        b3 = (e == 3 and col_num == 0) 
        b4 = (e == 1 and col_num == self.cols-1)
        # If so, return the 0 state.
        if b1 or b2 or b3 or b4: 
            return 0 
        
        # Get next state
        next_state = s + [-self.cols, 1, self.cols, -1][e]
                
        # Check to see if we've moved into a hole
        if next_state in self.holes:
            return 0
        
        return next_state
    
    
    def get_actions(self):
        return self.actions


    def get_states(self):
        return range(self.num_states)

    def get_state(self):
        return self.state

    def take_action(self, a):
        '''
        Take an action from a current state and return a new instance. 
        '''
        
        new_node = self.copy()
        s = self.state
        
        # Get the next state. 
        # If action taken from a hole or goal, return 0
        if s in self.holes + [0, self.goal]:
            next_state = 0
        # Otherwise, determine if the agent slipped and then get new state.
        else:
            trans = new_node.dynamics[s][a] # get trans info for state/action
            roll = np.random.uniform(0,1)
            cum_prob = np.cumsum(trans['prob'])
            n = np.sum(cum_prob < roll)
            next_state = trans['next_state'][n]
        
        new_node.state = next_state
        
        # Check to see if we are in a terminal state. 
        # Note that holes are not terminal, but the immeidately move to a terminal state. 
        if next_state in [0, self.goal]:
            new_node.terminal = True
     
        # Determine reward
        r = new_node.get_reward(s, next_state)
        
        # Update history
        new_node.actions_taken.append(a)
        new_node.rewards.append(r)
        new_node.path.append(next_state)
            
        return new_node
    
    
    def get_reward(self, cur_state, next_state):
        '''
        Determine the reward for moving from cur_state to next_state.
        '''
        
        # If the cur_state is 0, a hold, or the goal, then reward = 0. 
        if cur_state in self.holes + [0, self.goal]:
            return 0
        
        # Determine if agent has reached the goal. 
        if next_state == self.goal:
            return self.reward
        
        # Determine if agent has fallen into a goal or off the platform. 
        # Note: Penalty is recieved immediately if agent moves to hole. 
        # The next action will move the agent to 0, but no additional penalty 
        # will be applied due to first condition above. 
        if next_state in self.holes + [0]:
            return self.penalty
        
        # If we have not reached a terminal state, apply the step penalty. 
        return self.step_penalty
    
    def random_policy(self, random_state=None):
        '''
        Returns a randomly generated policy. 
        Currently used only be DPAgent. Should consider an alternative 
        approach that does not require this method. 
        '''
        if random_state is not None:
            np_state = np.random.get_state()
            np.random.seed(random_state)
        
        policy = {
            s:np.random.choice(self.actions)
            for s in range(self.num_states)
        }
        
        if random_state is not None:
            np.random.set_state(np_state)  
        
        return policy
    

    def display(self, size=1, fill=None, contents=None, 
                show_nums=True, show_path=False, show_fig=True):
        
        '''
        Displays the platform. 
        Attributes:
        size        Controls width of plot. 
        show_nums   Determines if cell numbers are displayed.
        show_path   Determines if the current path is displayed.
        policy      If a policy is provied, it will be displayed using arrows. 
        values      Determines cell shading. 
                    'slip' results in shading according to slip %
                    None results in no shading. 
                    A value function results in shading according to value. 
        show_fig    Controls if plt.show() is called. 
        '''
        
        # Determine shading for cells. 
        if fill is None:
            x = np.zeros(shape=(self.rows, self.cols))
            tsnorm = TwoSlopeNorm(vmin=0, vcenter=0.5, vmax=1)
            cm='binary'
        elif fill == 'slip':
            x = self.slip_prob[1:].reshape(self.rows, self.cols)
            tsnorm = TwoSlopeNorm(vmin=0, vcenter=0.5, vmax=1)
            cm = 'Greens'
        elif type(fill) == dict:    
            x = [fill.get(s,0) for s in range(1, self.num_states)]
            x = np.array(x).reshape(self.rows, self.cols)
            tsnorm = TwoSlopeNorm(vmin=-120, vcenter=0, vmax=120)
            cm = 'RdBu'
        elif type(fill) == np.ndarray:
            x = fill[1:].reshape(self.rows, self.cols)
            tsnorm = TwoSlopeNorm(vmin=-120, vcenter=0, vmax=120)
            cm = 'RdBu'
        else: 
            raise Exception('Argument provided for fill parameter is not understood.')
        
        # Calculate Sizes of various elements
        base = 3
        plot_width = base*size
        plot_height = plot_width * self.rows / self.cols
        num_size = 10 * base * size / self.cols
        glyph_size = 30 * base * size / self.cols
            
        if show_fig:
            plt.figure(figsize = [plot_width, plot_height])
            
        # Display platform. 
        plt.imshow(x, cmap=cm, norm=tsnorm)
        
        # Add contents and numbers to the cells. 
        # Glyphs include G for goal state, holes, and arrows for the policy.
        for i in range(1, self.num_states):
            
            glyph = None
            if i == self.goal: glyph = 'G' 
            elif i in self.holes: glyph = '■'
            
            if glyph is not None:
                plt.text((i-1)%self.cols, (i-1)//self.cols, glyph, ha='center', va='center',
                         fontdict={'size':glyph_size, 'weight':'bold'})
            
            elif type(contents) == str:  # Should only be 'slip'
                sp = self.slip_prob[i].round(2)
                plt.text((i-1)%self.cols, (i-1)//self.cols, f'{sp:<.2f}',
                         ha='center', va='center', 
                         fontdict={'size':glyph_size/2})
            
            elif type(contents) == dict:  # Should be a valid policy
                a = contents.get(i,0)
                glyph = {0:'↑', 1:'→', 2:'↓', 3:'←'}[a]
                plt.text((i-1)%self.cols, (i-1)//self.cols, glyph, ha='center', va='center',
                         fontdict={'size':glyph_size, 'weight':'bold'})

            
            if show_nums:
                txt = 'Start' if i == self.start else i
                plt.text(-0.45 + (i-1)%self.cols, -0.45 + (i-1)//self.cols, txt,
                         ha='left', va='top', 
                         fontdict={'size':num_size})
        
        # Display the path, if requested. 
        if show_path:
            for a, b in zip(self.path[:-1], self.path[1:]):
                if b != 0:
                    x1, y1 = (a-1)%self.cols, (a-1)//self.cols
                    x2, y2 = (b-1)%self.cols, (b-1)//self.cols
                    plt.plot([x1, x2], [y1, y2], c='k')
                elif a != 0:
                    x, y = (a-1)%self.cols, (a-1)//self.cols
                    plt.scatter(x, y, marker='x', c='k')    

        plt.xticks([])
        plt.yticks([])
        if show_fig:
            plt.show()
 
 
    def generate_episode(self, policy=None, epsilon=None, max_steps=None, 
                         show_steps=False, show_result=False, random_state=None):
        '''
        Generates an episode according to a policy. 
        '''
        
        # Fill out the policy, in case any states are missing. 
        if policy is not None:
            for s in range(self.num_states):
                a = np.random.choice(self.get_actions())
                policy[s] = policy.get(s, a)
            
        if max_steps == None:
            max_steps = float('inf')
            
        if random_state is not None:
            np_state = np.random.get_state()
            np.random.seed(random_state)
        
        n = 0
        node = self.copy()
        while node.terminal == False and n < max_steps:
            n += 1
            
            if policy is None:                      # No policy specified
                actions = self.get_actions()
                a = np.random.choice(actions)
            elif epsilon is None or epsilon == 0:   # Policy given, no exploration
                a = policy[node.state]
            else:                                   # Policy given, exploration
                roll = np.random.uniform(0,1)
                if roll < epsilon:
                    a = np.random.choice(self.actions)
                else:
                    a = policy[node.state]
                
            node = node.take_action(a)
            if show_steps:
                print(f'Taking action {a}. New state is {node.state}.')
            
        if random_state is not None:
            np.random.set_state(np_state)       
        
        if show_result:
            if node.state == node.goal:
                print('The agent reached the goal!')
            else:
                print('The agent failed to reach the goal.')
        
        return node
    

if __name__ == '__main__':
    
    fp = FrozenPlatform(4, 4, [0, 0.8], start=1, holes=[8, 13], random_state=391)
    pi1 = {0:0, 1:2, 2:2, 3:2, 4:3, 5:1, 6:1, 7:2, 8:0, 9:0, 10:1, 11:2, 12:2, 13:0, 14:1, 15:1, 16:0}

    fp.display()