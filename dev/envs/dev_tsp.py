import numpy as np 
import matplotlib.pyplot as plt


class TSP:
    
    def __init__(self, num_sites, random_state=None, is_copy=False):
        
        self.num_sites = num_sites

        # If this instance is a copy, then skip the steps below. 
        if is_copy: return
          
        # Generate site locations
        if random_state is not None:
            state = np.random.get_state()
            np.random.seed(random_state)
            
        self.sites = np.random.uniform(0, 100, size=(num_sites, 2)).astype(int)
            
        # Calculate Pairwise Distances
        self.distances = np.linalg.norm(
            np.expand_dims(self.sites, axis=1) - np.expand_dims(self.sites, axis=0),
            axis=-1
        ).round(1)
        
        # Record path information
        self.unvisited = set(range(1,num_sites))
        self.path = [0]
        self.cost = 0
        
        # Restore the NumPy state, if changed
        if random_state is not None:
            np.random.set_state(state)

    
    def __lt__(self, other):
        '''
        Implements "less than" operator for class. Required for priority queue. 
        '''
        return self
    
    
    def copy(self):
        '''
        Return a copy of this instance. 
        Sites and distances are referenced, not copied.
        '''
        new_node = TSP(self.num_sites, is_copy=True)
        new_node.sites = self.sites             # This is reference, not a copy
        new_node.distances = self.distances     # This is reference, not a copy
        
        new_node.unvisited = {*self.unvisited}  # Copy unvisited set
        new_node.path = [*self.path]            # Copy path list
        new_node.cost = self.cost               # Set cost
        return new_node
    
    
    def display(self, show_path=True, labels=True, figsize=[5,5], show_plot=True):
        '''
        Displays scatter plot of sites. 
        Has option to display current path. 
        '''
        
        if show_plot:
            plt.figure(figsize=figsize)
           
        if labels: 
            for i in range(self.num_sites):
                plt.text(self.sites[i,0]+1.5, self.sites[i,1]+3, s=i)
        
        if show_path:
            pts = self.sites[self.path, :]
            plt.plot(pts[:, 0], pts[:,1], linewidth=1, c='darkgrey')
   
        plt.scatter(self.sites[1:,0], self.sites[1:,1], zorder=10)
        plt.scatter(self.sites[0,0], self.sites[0,1], zorder=10)
        
        plt.xticks([])
        plt.yticks([])
        
        plt.xlim([-10, 110])
        plt.ylim([-10, 110])
        
        if show_plot:
            plt.show()
    
    
    def check_solved(self):
        '''
        Checks to see if the current state represents a solution. 
        '''
        return len(self.unvisited) == 0
    
    
    def soln_info(self):
        '''
        Used to report path and cost in search output. 
        '''
        msg = f'Solution Path: {self.path}\n'
        msg = f'Solution Cost: {self.path_cost()}'
        return(msg)
    
    
    def get_state_id(self):
        '''
        Returns None since there is no possibility of returning to a previous state. 
        '''
        return None
            
            
    def get_actions(self):
        '''
        Return set of unvisited sites. 
        '''
        return self.unvisited
    
    
    def take_action(self, site):
        '''
        Add requested site to path.
        If adding last unvisited site, path will close. 
        '''
        new_node = self.copy()
        current = new_node.path[-1]
        new_node.unvisited.remove(site)
        new_node.path.append(site)
        d = self.distances[current, site]
        new_node.cost = round(new_node.cost + d, 1)
        
        # Check if there are still unvisited sites. If not, close path.
        if len(new_node.unvisited) == 0:
            new_node.path.append(0) 
            d = self.distances[site, 0]
            new_node.cost = round(new_node.cost + d, 1)
            
        return new_node


    def random_walk(self, steps=None, random_state=None):
        '''
        Generates a random walk through the sites. 
        '''
        if steps == None:
            steps = self.num_sites - 1 
       
        if random_state is not None:
            np_state = np.random.get_state()
            np.random.seed(random_state)
       
        new_node = self.copy()
        for i in range(steps):
            actions = new_node.get_actions()
            a = np.random.choice(list(actions))
            new_node = new_node.take_action(a)
       
        if random_state is not None:
            np.random.set_state(np_state)
        return new_node
                
                
    def path_cost(self):
        '''
        Returns the path cost. 
        '''
        return self.cost
    
    
    def heuristic(self, alg):
        '''
        Wrapper function for heuristics. 
        '''
        if alg == 'AST': return self.ast_heuristic()
        if alg == 'GBF': return self.gbf_heuristic()


    def ast_heuristic(self):
        '''
        Provides lower bound for cost required to complete the path.
        For each target site, finds min distance from possible source sites. 
        Target sites include goal site and all unvisited sites. 
        Source sites include cur site and all unvisited sites. 
        '''
        if self.check_solved():
            return 0

        # Selected rows will define potential source sites
        # Selected cols will define potential target sites
        sel_rows = list(self.unvisited) + [self.path[-1]]
        sel_cols = list(self.unvisited) + [0]
        
        # Select relevant distances. Replace 0 distances with inf
        dist = self.distances[sel_rows, :][:, sel_cols]
        dist = np.where(dist == 0, np.inf, dist)
        dist[-1,-1] = np.inf
        
        # For each target, find dist to nearest source. Sum results. 
        dist_to_nearest = dist.min(axis=0)
        h_value = dist_to_nearest.sum()
        
        return round(h_value, 1)


    def gbf_heuristic(self):
        '''
        Heuristic used for greedy algorithm. 
        '''
        # Calculate distance travelled in previous step. 
        a = self.path[-2]
        b = self.path[-1]
        d = self.distances[a, b]
        
        # Selected rows will define potential source sites
        # Selected cols will define potential target sites
        sel_rows = list(self.unvisited) + [self.path[-1]]
        sel_cols = list(self.unvisited) + [0]
        
        # Select relevant distances.
        dist = self.distances[sel_rows, :][:, sel_cols]
        
        # Find largest distance between any remaining source and target
        max_dist = dist.max() * (1 + len(self.unvisited))
        
        return round(max_dist + d, 1)
            
    
if __name__ == '__main__':
    tsp = TSP(num_sites=12, random_state=3)
    tsp = tsp.random_walk(steps=3, random_state=1)
    tsp.display()
    
    