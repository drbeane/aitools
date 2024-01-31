import numpy as np

class Knapsack:
    
    def __init__(self, num_items, capacity, random_state=None, is_copy=False):
        
        self.num_items = num_items
        self.capacity = capacity
        self.n_constraints = len(capacity)
        
        if is_copy:
            return
        
        self.inventory = np.zeros(num_items).astype(int)
        self.total_weight = np.zeros(self.n_constraints).astype(int)
        self.total_value = 0
        
        
        if random_state is not None:
            np_state = np.random.get_state()
            np.random.seed(random_state)
            
        self.beta_dist()
        
        if random_state is not None:
            np.random.set_state(np_state)
    
    
    def beta_dist(self):
        # Determine Item Weights
        wts = np.random.beta(a=2, b=4, size=(self.num_items, self.n_constraints))
        self.weights = np.ceil(10 * wts).astype(int)
        
        # Determine Item Values
        k = self.weights.sum(axis=1)
        k = k / (k.max() + 1e-6)
        alpha = 3
        beta = 8*alpha*(1-k)/k
        vals = np.random.beta(a=alpha, b=beta, size=self.num_items)
        self.values = np.ceil(50 * vals).astype(int)
        
    
    def copy(self):
        temp = Knapsack(self.num_items, self.capacity, is_copy=True)
        temp.inventory = np.array(self.inventory)
        temp.total_weight = np.array(self.total_weight)
        temp.total_value = self.total_value
        
        temp.weights = self.weights
        temp.values = self.values
        
        return temp
        
        
    def display(self, flags=''):
        if 'w' in flags:
            print(f'Item Weights:\n{self.weights.T}\n')
        if 'v' in flags:
            print(f'Item Values:\n{self.values}\n')
        if 'i' in flags:
            print(f'Current Inventory:\n{self.inventory}\n')
        print('Knapsack Capacity:', self.capacity)
        print('Current Weight:   ', self.total_weight)
        print('Current Value:    ', self.total_value)
        print('Is Valid Solution:', self.check_solved())
        print()


    def get_actions(self):
        avail_cap = self.capacity - self.total_weight
        item_fit_flag = (self.weights <= avail_cap).prod(axis=1)
        unused_flag = (self.inventory == 0)
        valid_option_flag = unused_flag & item_fit_flag
        options = np.where(valid_option_flag)[0]
        
        return options

        
    def take_action(self, item):
        new_state = self.copy()
        new_state.inventory[item] = 1
        new_state.total_value += self.values[item]
        new_state.total_weight += self.weights[item, :]
        return new_state

    
    def check_solved(self):
        
        avail_cap = self.capacity - self.total_weight
        
        # Check to see if we are over capacity
        if avail_cap.min() < 0:
            return False
        
        # Check to see if there are available actions
        if len(self.get_actions()) == 0:
            return True
        return False

        
    def random_fill(self, verbose=True):
        new_state = self.copy()
        return new_state
    
    
    def set_state(self, inv):
        new_state = self.copy()
        
        new_state.inventory = inv
        new_state.total_weight = (inv.reshape(-1,1) * new_state.weights).sum(axis=0)
        new_state.total_value = (inv * new_state.values).sum()
       
        return new_state
    
    
    def reset(self):
        self.inventory = np.zeros(self.num_items).astype(int)
        self.total_weight = np.zeros(self.n_constraints).astype(int)
        self.total_value = 0
              

    def get_gen_alg_params(self):
        alleles = [0,1]
        soln_size = self.num_items
        return alleles, soln_size
    
    
    def evaluate_fitness(self, specimen):
        
        total_weight = (specimen.reshape(-1,1) * self.weights).sum(axis=0)
        
        # Check to see if we are over capacity
        avail_cap = self.capacity - total_weight
        if avail_cap.min() < 0:
            return 0
        
        value = (specimen * self.values).sum()
        return int(value)

    def lp_solver(self):
        import pulp as pl
        
        pl_vars = [[] for i in range(self.n_jobs+1)]        # List for storing pulb variable names: v_1_1, v_1_2, ...  v_w_j
        costs = []                  # List for containing expressions representing worker costs: 4*v_1, 8*v_2, ...
        total_skill = [[] for i in range(self.n_jobs+1)]    # List of lists. Each list represents the skill of workers assigned to a job. 
        
        for w in range(self.n_workers):                                  # Loop over all workers
            for j in range(self.n_jobs+1):                               # Loop over all jobs
                v = pl.LpVariable(f'v_{w}_{j}', 0, 1, pl.LpInteger)      # Create variable
                pl_vars[j].append(v)                                        # Add variable to list
                costs.append(v * self.worker_costs[w])                  # Add worker cost to list         
                total_skill[j].append(v * self.worker_skill[j,w])  # Add worker skill to list for each job 
        
        problem = pl.LpProblem('JobAssignment', pl.LpMinimize)     # Define pulp problem.
        problem += sum(costs)                                      # Add objective      
        
        # Add Job Requirement Constraints
        for j in range(self.n_jobs+1):                             
            problem += sum(total_skill[j]) >= self.job_reqs[j]    
        
        # Add one job per worker constraints. 
        for w in range(self.n_workers):
            worker_col = [row[w] for row in pl_vars]  # Extract col of vars relating to one worker 
            problem += sum(worker_col) <= 1
        
        status = problem.solve()

        # Create Solution State
        new_state = self.copy()
        for j in range(self.n_jobs+1):
            for w, v in enumerate(pl_vars[j]):
                if pl.value(v) == 1:
                    new_state = new_state.take_action((w,j))
  
        print(new_state.soln_info())
        return new_state



# Note: Search techniques don't work well for this problem. There is not a good notion of path cost. 

    
if __name__ == '__main__':
    
    np.random.seed(1)
    state = Knapsack(num_items=10, capacity=[20,24])
    state.display()
    
    sp = np.array([0, 1, 0, 1, 1, 0, 1, 0, 1, 0])
    print(state.evaluate_fitness(sp))
    
    #state = state.random_fill()
    
    #state.display()