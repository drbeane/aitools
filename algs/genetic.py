import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


class GeneticAlgorithm:
    
    def __init__(self, env, pop_size, init_zero=False, random_state=None):
    
        self.env = env
        self.pop_size = pop_size
        self.rr = 0.4
        self.mr = 1e-6
        self.transformation = None
        self.epsilon = 0
        self.crossover = 'uniform'
        self.np_random_state = None
        self.generations = 0
        
        if random_state is not None:
            np_state = np.random.get_state()
            np.random.seed(random_state)
        
        self.alleles, self.soln_size = env.get_gen_alg_params()
        
        pop_shape = (self.pop_size, self.soln_size)
        if init_zero:
            self.population = np.zeros(shape=pop_shape).astype(int)
        else:
            self.population = np.random.choice(self.alleles, size=pop_shape).astype(int)

        self.scores = np.zeros(pop_size)
        self.evaluate_population()
        
        self.history = {'min': [], 'q1':[], 'median': [], 'q3':[], 'max': []}
        
        if random_state is not None:
            self.np_random_state = np.random.get_state()
            np.random.set_state(np_state)    
            
    
    def evaluate_population(self):
        for i in range(self.pop_size):
            specimen = self.population[i, :]
            fitness = self.env.evaluate_fitness(specimen)
            self.scores[i] = fitness


    def roulette_selection(self, sample_size):
        values = np.array(self.scores)
        if self.transformation == 'square':
            values = values ** 2
        values += self.epsilon
        total = values.sum()
        probs = values / total
        sel = np.random.choice(range(self.pop_size), size=sample_size, replace=False, p=probs)
        return sel


    def generate_children(self):
        num_parents = int(self.pop_size * self.rr)
        if num_parents % 2 == 1: num_parents += 1
        parents = self.roulette_selection(num_parents)
        np.random.shuffle(parents)
        num_children = num_parents // 2
        children = np.zeros(shape=(num_children, self.soln_size)).astype(int)
        for i in range(num_children):
            p1 = self.population[parents[2 * i], :]
            p2 = self.population[parents[2 * i + 1], :]
            children[i] = self.make_child(p1, p2)
        
        return children


    def make_child(self, p1, p2):
        if self.crossover == 'uniform':
            mask = np.random.choice([0, 1], size=self.soln_size)
            child = p1 * mask + p2 * (1 - mask)
        if self.crossover == 'split':
            cut_point = np.random.choice(range(1, self.env.soln_size-1))
            child = np.hstack([p1[:cut_point], p2[cut_point:]])
        return child
    
    
    def run_evolution(self, num_generations, rr, mr, crossover='uniform', transformation=None, 
                      epsilon=1e-6, update_rate=None, show_progress=False):
        
        self.rr = rr
        self.mr = mr
        self.crossover = crossover
        self.transformation = transformation
        self.epsilon = epsilon

        n = len(str(num_generations))

        if self.np_random_state is not None:
            temp_random_state = np.random.get_state()
            np.random.set_state(self.np_random_state)

        best_score = float('-inf')
        best_soln = None

        rng = tqdm(range(num_generations)) if show_progress else range(num_generations)
        for i in rng:
            children = self.generate_children()
            
            n_survivors = self.pop_size - len(children)
            surv_idx = self.roulette_selection(n_survivors)
            survivors = self.population[surv_idx, :]
            
            mutation_mask = np.random.choice([0,1], size=(n_survivors, self.soln_size),
                                             p=[1 - self.mr, self.mr])
            mutations = np.random.choice(self.alleles, size=(n_survivors, self.soln_size))

            survivors = survivors * (1 - mutation_mask) + mutations * mutation_mask
            
            self.population = np.vstack([survivors, children])
            self.evaluate_population()

            max_score = self.scores.max()
            if max_score > best_score:
                best_score = int(max_score)
                idx = self.scores.argmax()
                best_soln = self.population[idx,:].copy()
                gen = self.generations + i + 1
                print(f'Generation {gen:0{n}}: New best score found! Current best score = {best_score}')
                

            perc = np.percentile(self.scores, q=[0.25, 0.5, 0.75]).astype(int)
            minm = self.scores.min().astype(int)
            maxm = self.scores.max().astype(int)

            self.history['min'].append(minm)
            self.history['q1'].append(perc[0])
            self.history['median'].append(perc[1])
            self.history['q3'].append(perc[2])
            self.history['max'].append(maxm)

            if update_rate is not None:
                if (i+1) % update_rate == 0:
                    gen = self.generations + i + 1
                    print(f'Generation {gen:0{n}}: Q1={perc[0]}, Med={perc[1]}, Q3={perc[2]}, Max={maxm}')


        if self.np_random_state is not None:
            self.np_random_state = np.random.get_state()   # Store the current random state. 
            np.random.get_state(temp_random_state)         # Set state back as it was before method was called. 

        self.generations += num_generations

        #path_idx = self.scores.argmax()
        #path = self.population[path_idx,:]
        
        return self.env.set_state(best_soln)

    def plot_history(self, show_min=False):
        
        for k in self.history.keys():
            if k == 'min' and show_min == False:
                continue
            plt.plot(self.history[k], label=k)
        plt.legend()
        plt.xlabel('Generation')
        plt.ylabel('Score')
        plt.show()


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


if __name__ == '__main__':
    
    import sys
    sys.path.append('algorithms/')
    sys.path.append('environments/')
    from aitools.envs import Knapsack
    
    ks = Knapsack(num_items=20, capacity=[40, 30], random_state=1)

    ga = GeneticAlgorithm(env=ks, pop_size=1000, init_zero=True, random_state=1)

    soln = ga.run_evolution(150, rr=0.4, mr=1e-3, update_rate=20)
    