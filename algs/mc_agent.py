import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from matplotlib.cm import get_cmap


class MCAgent:
    
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
        self.sa_visits = {}
        
        # Store random state. This will be used for all methods as well. 
        self.random_state = random_state
        self.np_random_state = None
        if random_state is not None:
            np_state = np.random.get_state()               # Get current np random state
            np.random.seed(random_state)                   # Set seed to value provided
            self.np_random_state = np.random.get_state()   # Record new np random state
            np.random.set_state(np_state)                  # Reset old np random state
     
        
    def copy(self):
        new_agent = MCAgent(self.env, self.gamma, self.policy)
        new_agent.V = self.V.copy()
        new_agent.s_visits = self.s_visits.copy()
        new_agent.V_history = self.V_history.copy()
        new_agent.s_visits = self.sa_visits.copy()
        new_agent.Q = self.Q.copy()
        new_agent.sa_visits = self.sa_visits.copy()
        new_agent.np_random_state = self.np_random_state
        return new_agent
    
        
    def evaluate_policy(self, episodes, alpha=None, exploring_starts=False, 
                        max_steps=None, track_history=True, show_progress=True):
        '''
        Implements first-visit MC Prediction. 
        '''
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
            ep = node.generate_episode(self.policy, max_steps=max_steps)
            
            # Calculate Returns for each step. 
            k = len(ep.rewards)
            d = self.gamma ** np.arange(k)
            rew = np.array(ep.rewards)
            G = [np.sum(rew[j:] * d[:k-j]) for j in range(k)]
            
            # Iterate over the steps in the episode, updating the value function. 
            visited_states = set()
            for s, g in zip(ep.path[:-1], G):
                if s not in visited_states:
                    visited_states.add(s)
                    n = self.s_visits.get(s,0) + 1
                    v = self.V.get(s,0)
                    
                    self.s_visits[s] = n
                    
                    if alpha is None:
                        self.V[s] = v + (g - v)/n  
                    else:
                        self.V[s] = v + (g - v) * alpha                    
            
            if track_history:
                self.V_history.append(self.V.copy())
                        
        if self.np_random_state is not None:
            self.np_random_state = np.random.get_state()  # Store current state for future use.
            np.random.set_state(np_state)                 # Return state to previous value
        
        return 
 
        
    def control(self, episodes, epsilon, alpha=None, exploring_starts=False, 
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
                start = np.random.choice(states)
                node = self.env.reset(start)
            else:
                node = self.env.copy()
                
            # Generate an episode. 
            ep = node.generate_episode(self.policy, epsilon=epsilon, max_steps=max_steps)
            
            # Calculate Returns for each step. 
            k = len(ep.rewards)
            d = self.gamma ** np.arange(k)
            rew = np.array(ep.rewards)
            G = [np.sum(rew[j:] * d[:k-j]) for j in range(k)]
            
            # Iterate over steps in episode. 
            # Update Q, V, and pi at each step. 
            visited_pairs = set()
            for s, a, g in zip(ep.path[:-1], ep.actions_taken, G):
                if (s,a) not in visited_pairs:
                    visited_pairs.add((s,a))
                    
                    n = self.sa_visits.get((s,a), 0) + 1
                    q = self.Q.get((s,a), 0)
                    
                    self.sa_visits[s,a] = n
                    
                    if alpha is None:
                        self.Q[s,a] = q + (g - q)/n
                    else:
                        self.Q[s,a] = q + (g - q) * alpha
                    
                    # Get largest value for Q(s,*), and corresponding action
                    best_q = float('-inf')
                    best_a = None
                    for a_ in node.get_actions():
                        temp_q = self.Q.get((s,a_), 0)
                        if temp_q > best_q:
                            best_q = temp_q
                            best_a = a_
                        
                    self.policy[s] = best_a
                    self.V[s] = best_q
            
            if track_history:
                self.V_history.append(self.V.copy())
         
        if self.np_random_state is not None:
            self.np_random_state = np.random.get_state()  # Store current state for future use.
            np.random.set_state(np_state)                 # Return state to previous value 
            
            
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
 
 
    def V_(self, ns=None):
        if ns is None:
            ns = max(list(self.V.keys())) + 1
        return np.array([self.V.get(s,0) for s in range(ns)])
    