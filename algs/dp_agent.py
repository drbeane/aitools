import numpy as np
import matplotlib.pyplot as plt
import math
import time

class DPAgent:
    
    def __init__(self, env, gamma, policy=None, random_state=None):
        self.env = env.copy()
        self.gamma = gamma
        
        if policy is None:
            self.policy = self.env.random_policy(random_state)
        else:
            self.policy = policy
        
        self.V = {s:0 for s in range(env.num_states)}
        self.eval_steps = 0
        
        self.converged = False
        self.history = {
            'V':[], 
            'policy':[self.policy]
        }
        

    def evaluate_policy(self, threshold=1e-6, max_iter=None, report=True):
                
        if max_iter is None:
            max_iter = float('inf')
        
        n = 0
        while n < max_iter:
            n += 1
            max_diff = 0
            for s in self.env.get_states():
                oldV = self.V[s]
                a = self.policy[s]
                
                trans = self.env.dynamics[s][a]
                prob = trans['prob']
                next_state = trans['next_state']
                newV = 0
                for p, s_ in zip(prob, next_state):
                    r = self.env.get_reward(s, s_)
                    newV += p * (r + self.gamma * self.V[s_])
                    
                self.V[s] = newV
                max_diff = max(max_diff, abs(newV - oldV))
            
            if max_diff < threshold:
                break                
        
        self.eval_steps += n
        
        if report:
            print(f'Policy evaluation required {n} iterations.')
        
        
    def policy_improvement(self):
        
        new_policy = {}
        for s in self.env.get_states():
            
            best_action = None
            best_return = float('-inf')
            for a in self.env.get_actions():         # Loop over all actions
                exp_return = 0
                trans = self.env.dynamics[s][a]
                for p, s_ in zip(trans['prob'], trans['next_state']):
                    r = self.env.get_reward(s, s_)
                    exp_return += p * (r + self.gamma * self.V[s_])
                
                if exp_return > best_return:
                    best_action = a
                    best_return = exp_return
            
            new_policy[s] = best_action
        
        stable = (new_policy == self.policy)
        self.policy = new_policy
        
        return stable
        
            
    def policy_iteration(self, threshold=1e-6, track_history=True, report=True, updates=None):
        
        if self.converged == True:
            print('Policy is already optimal.')
            return
        
        t0 = time.time()
        policy_stable = False
        n = 0
        while policy_stable == False:
            n += 1
            old_policy = self.policy.copy()  
            # Copy is not strictly necessary in line above. 
            # Policy improvement creates a new policy dictionary. 
            
            self.evaluate_policy(report=False, threshold=threshold)
            policy_stable = self.policy_improvement()
        
            if updates is not None:
                if n % updates == 0:
                    changed = len(
                        [s for s in self.env.get_states()
                         if old_policy[s] != self.policy[s]]
                    )
                    print(f'Iteration {n}: {changed} state/actions updated.')
            
            if track_history:
                self.history['V'].append(self.V.copy())
                self.history['policy'].append(self.policy)
        
        t = time.time() - t0
        if report:
            print(f'Policy Iteration took {n} steps and {self.eval_steps} evaluation sweeps to converge.')
            print(f'Total time required was {t:.2f} seconds.')        
        self.converged = True
    
    
    def value_iteration(self, threshold=1e-6, track_history=True, report=True, updates=None):
        
        if self.converged == True:
            print('Policy is already optimal.')
            return 
        
        t0 = time.time()
        
        n = 0
        while True:
            n += 1
            
            old_V = self.V.copy()
            self.evaluate_policy(report=False, max_iter=1)
            self.policy_improvement()
            
            if track_history:
                self.history['V'].append(self.V.copy())
                self.history['policy'].append(self.policy)
                
            #max_diff = np.abs(old_V - self.V).max()
            max_diff = 0
            for s in range(self.env.num_states):
                max_diff = max(max_diff, abs(old_V[s] - self.V[s]))
            
            
            if updates is not None:
                if n % updates == 0:
                    print(f'Iteration {n}: Max Diff = {max_diff:<.6f}')
            
            if max_diff < threshold:
                break
            
        t = time.time() - t0  
        if report: 
            print(f'Value Iteration took {n} steps to converge.')     
            print(f'Total time required was {t:.2f} seconds.')  
        self.converged = True


    def show_history(self, cols=6, size=0.5, **kwargs):
         
        n = len(self.history['V'])
         
        cols = min(cols, n)
        rows = math.ceil(n / cols)
        
        if size is not None:
            plt.figure(figsize=[3.5*size*cols, 3.5*size*rows])
        
        for i in range(n):
            plt.subplot(rows, cols, i+1)
            self.env.display(
                show_fig=False, 
                contents=self.history['policy'][i],
                fill=self.history['V'][i],
                size=size,
                show_nums=False,
                **kwargs
            )
        plt.show()

    
    def report(self):
        
        n = len(self.history['V'])
        
        for i in range(n):
            
            pi_0 = self.history['policy'][i]
            pi_1 = self.history['policy'][i+1]
            
            changed = [s for s in self.env.get_states() if pi_0[s] != pi_1[s]]
            
            print(f'Update {i+1} - {len(changed)} state/action pairs changed')