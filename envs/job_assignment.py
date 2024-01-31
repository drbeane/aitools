import numpy as np


class JobAssignment:

    def __init__(self, n_jobs, n_workers, max_skill=6, random_state=None, copy=False):
        '''
        '''

        self.n_jobs = n_jobs
        self.n_workers = n_workers
        self.assignments = np.zeros(n_workers).astype(int)
        self.total_skill = np.zeros(n_jobs+1).astype(int)
        self.total_cost = 0

        if copy == False:
            rand_state = np.random.get_state()
            if random_state is not None:
                np.random.seed(random_state)
            
            self.job_reqs = np.random.choice(range(1,17), size=n_jobs+1)
            self.job_reqs[0] = 0
            self.worker_skill = np.random.choice(range(1,max_skill+1), size=(n_jobs+1, n_workers))
            self.worker_skill[0,:] = 0
            self.worker_costs = np.random.choice(range(1,10), size=n_workers)
            np.random.get_state(rand_state)


    def __lt__(self, other):
        return self


    def copy(self):
        new_state = JobAssignment(self.n_jobs, self.n_workers, copy=True)
        new_state.assignments = np.array(self.assignments)
        new_state.total_skill = np.array(self.total_skill)
        new_state.total_cost = self.total_cost
        new_state.job_reqs = self.job_reqs
        new_state.worker_skill = self.worker_skill
        new_state.worker_costs = self.worker_costs
        return new_state


    def get_state_id(self):
        return self.assignments.tostring()


    def random_assign(self):
        self.total_skill = self.total_skill*0
        self.assignments = np.random.choice(range(self.n_jobs+1), self.n_workers)
        for w, j in enumerate(self.assignments):
            self.total_skill[j] += self.worker_skill[j, w]


    def random_assignment(self):
        return np.random.choice(range(self.n_jobs+1), self.n_workers)


    def path_cost(self):
        return self.total_cost


    def display(self, flags='rcsat'):
        if 'r' in flags:
            print(f'Job Requirements: {self.job_reqs}\n')
        if 'c' in flags:
            print(f'Worker Costs:     {self.worker_costs}\n')
        if 's' in flags:
            print(f'Worker Skills:    \n{self.worker_skill}\n')
        if 'a' in flags:
            print('Assignments:   [ ', end='')
            for a in self.assignments:
                if a == 0:
                    print('_ ', end='')
                else:
                    print(a, end=' ')
            print(']\n')
        if 't' in flags:
            print(f'Total cost: {self.total_cost}')
        print()


    def soln_info(self):
        msg = 'Assignments: [ '
        for a in self.assignments:
            if a == 0:
                msg += '_ '
            else:
                msg += f'{a} '
        msg += ']\n'                
        
        msg += f'Job Reqs:    {self.job_reqs}\n'
        msg += f'Total Skill: {self.total_skill}\n'
        msg += f'Total Cost:  {self.path_cost()}'
        return msg
        
        
    def check_solved(self):        
        return np.all(self.total_skill >= self.job_reqs)


    def get_actions(self):
        workers = np.argwhere(self.assignments == 0).reshape(-1,)
        jobs = range(1, self.n_jobs+1)
        actions = [(w,j) for w in workers for j in jobs]
        return actions

    def get_path(self):
        return None

    def take_action(self, a):
        w, j = a
        new_state = self.copy()
        new_state.assignments[w] = j
        new_state.total_skill[j] += self.worker_skill[j, w]
        new_state.total_cost += self.worker_costs[w]
        return new_state


    def heuristic(self, **kwargs):
        unassigned = np.argwhere(self.assignments == 0).reshape(-1,)
        if len(unassigned) == 0:
            return 0
        ua_skills = self.worker_skill[1:, unassigned]
        ua_costs = self.worker_costs[unassigned]
        ua_rates = ua_costs / ua_skills 
        best_rates = np.min(ua_rates, axis=1)
        diff = self.job_reqs - self.total_skill 
        diff = np.where(diff > 0, diff, 0)[1:]
        h = np.sum(diff * best_rates)
        return  h

    '''
    GA Functions below
    '''
    def get_gen_alg_params(self):
        alleles = range(self.n_jobs+1)
        soln_size = self.n_workers
        return alleles, soln_size
    
    
    def evaluate_fitness(self, schedule):
        
        # Determine Skill 
        skill_by_worker = self.worker_skill[schedule, np.arange(self.n_workers)]
        skill_by_job = np.bincount(schedule, weights=skill_by_worker, minlength=self.n_jobs+1)
        reqs_met = np.all(skill_by_job >= self.job_reqs)
        if reqs_met == False:
            return 0
        
        # Determine Cost
        sel = (schedule != 0)
        total_cost = self.worker_costs[sel].sum()
        worst_case = 10 * self.n_workers
        fitness = worst_case - total_cost
        
        return fitness


    def set_state(self, assignments):
        new_state = self.copy()
        for w,j in enumerate(assignments):
            if j != 0:
                new_state = new_state.take_action((w,j))
        return new_state
    
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
    import random

    ja = JobAssignment(n_jobs=6, n_workers=10)

    print(ja.worker_skill)
    print()
    
    #ja.random_assign()

    for i in range(4):
        actions = ja.get_actions()
        a = random.choice(actions)
        ja = ja.take_action(a)

    ja.display()

    print('Solved:', ja.check_solved())

    print('Heuristic', ja.heuristic())