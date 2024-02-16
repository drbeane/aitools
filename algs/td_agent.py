import numpy as np
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import gc

class TDAgent:
    
    def __init__(self, env, gamma, policy=None, random_state=None):
        
        self.env = env.copy()
        self.gamma = gamma
        
        # Store the policy, or generate a random one if no policy is provided. 
        if policy is None:
            self.policy = {}
        else:
            self.policy = policy.copy()
        
        self.V = {env.state:0}
        self.s_visits = {}
        self.V_history = []

        self.Q = {}
            
        # Store random state. This will be used for all methods as well. 
        self.random_state = random_state
        self.np_random_state = None
        if random_state is not None:
            np_state = np.random.get_state()               # Get current np random state
            np.random.seed(random_state)                   # Set seed to value provided
            self.np_random_state = np.random.get_state()   # Record new np random state
            np.random.set_state(np_state)                  # Reset old np random state
    
    
    def copy(self):
        new_agent = TDAgent(self.env, self.gamma, self.policy)
        new_agent.V = self.V.copy()
        new_agent.s_visits = self.s_visits.copy()
        new_agent.V_history = None if self.V_history is None else self.V_history.copy()
        new_agent.Q = self.Q.copy()
        new_agent.np_random_state = self.np_random_state
        return new_agent    
    
    
    def q_learning(self, episodes, epsilon, alpha, exploring_starts=False, 
                   max_steps=None, track_history=True, show_progress=True):
            
        if max_steps == None:
            max_steps = float('inf')
        
        if self.np_random_state is not None:
            np_state = np.random.get_state()
            np.random.set_state(self.np_random_state)
        
        rng = tqdm(range(episodes)) if show_progress else range(episodes)
        for i in rng:
            
            if exploring_starts:
                states = [*self.V.keys()]
                idx = np.random.choice(len(states))
                start = states[idx]
                node = self.env.reset(set_start=start)
            else:
                node = self.env.reset()

            
            # Generate an episode. 
            n = 0
            while node.terminal == False and n < max_steps:
                
                #node.display()
                
                n += 1
                
                # Record current state
                s = node.get_state()
                
                # Get actions available in current state. 
                available_actions = node.get_actions()
                
                # Get next action
                roll = np.random.uniform(0,1)
                if roll < epsilon:   
                    idx = np.random.choice(len(available_actions))
                    a = available_actions[idx]
                    #a = np.random.choice(self.env.actions)
                else:                        
                    a = self.policy.get(s,None)
                    #if a is None:
                    if a not in available_actions:
                        idx = np.random.choice(len(available_actions))
                        a = available_actions[idx]
                        #a = np.random.choice(self.env.actions)
                    
                
                # Get next node and state
                node = node.take_action(a)
                s_ = node.get_state()
                
                # Get reward 
                r = node.rewards[-1]
                
                # Update Q
                q = self.Q.get((s,a), 0)
                q_est = r + self.gamma * self.V.get(s_, 0)
                
                self.Q[(s,a)] = q + alpha * (q_est - q)
                #self.V[s] = max(self.V.get(s, 0), self.Q[(s,a)])
                
                # Get largest value for Q(s,*), and corresponding action
                best_q = float('-inf')
                best_a = None
                for a_ in available_actions:
                    temp_q = self.Q.get((s,a_), 0)
                    if temp_q > best_q:
                        best_q = temp_q
                        best_a = a_
                
                #print(self.policy)   
                self.policy[s] = best_a
                self.V[s] = best_q

            if track_history:
                self.V_history.append(self.V.copy())   
             
            if i % 1000 == 0:
                gc.collect() 
                
        if self.np_random_state is not None:
            self.np_random_state = np.random.get_state()  # Store current state for future use.
            np.random.set_state(np_state)                 # Return state to previous value 
    
    def V_(self, ns=None):
        if ns is None:
            ns = max(list(self.V.keys())) + 1
        return np.array([self.V.get(s,0) for s in range(ns)])
    
    def plot_v_history(self, figsize=[4,6], states=None, target=None):
        if states is None:
            states = list(self.V.keys())
            
        k = len(states)
        vhist = np.array(self.V_history)
        plt.figure(figsize=figsize)
        cmap = get_cmap('nipy_spectral')
        cstep = 0.99 / k
        
        for i, s in enumerate(states):
            vh = [v.get(s,0) for v in self.V_history]
            
            plt.plot(vh, label=s, c=cmap(i*cstep), lw=0.8, alpha=0.8)
            if target is not None:
                plt.scatter(vhist.shape[0], target[s], color=cmap(i*cstep), zorder=5)
        plt.xlabel('Number of Episodes')
        plt.ylabel('Estimate of State Value Function')
        plt.title('Evolution of State Value Function')
        plt.legend(bbox_to_anchor=[1,1])
        plt.show()
     
if __name__ == '__main__':
    import sys
    sys.path.append('environments/')
    from frozen_platform import *
    
    fp = FrozenPlatform(4, 4, [0.1, 0.4], holes=0, random_state=1)

    td = TDAgent(env=fp, gamma=1, random_state=1)
    td.q_learning(episodes=10, epsilon=0.001, alpha=0.01, show_progress=False, max_steps=200, exploring_starts=False)  