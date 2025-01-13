import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay

class AltRoutePlanning:
    
    def __init__(self, num_sites, q=0.2, start=None, goal=None, random_state=None, is_copy=False):
        
        self.num_sites = num_sites
        
        # If this instance is being created as a copy, skip remaining steps
        if is_copy: return
        # Lines above executed for all nodes, copy or otherwise
        # Lines below are executed only for new nodes
        
        self.state = 0 if start is None else start 
        self.goal = num_sites-1 if goal is None else goal
        self.start = self.state
        
        if random_state is not None:
            np_state = np.random.get_state()  # Store NumPy random state to restore later
            np.random.seed(random_state)      # Set the seed
        
        
        while True:  # Loop until a solvable example is obtained. 
            # Generate site locations
            #num_gen = int(num_sites/(1-q))
            #k = int(num_gen**0.5)
            #site_arrays = [
            #    np.array([(0,0), (0,100), (100,0), (100,100)]),
            #    [(0,x) for x in np.random.uniform(0, 100, k).astype(int)],
            #    [(100,x) for x in np.random.uniform(0, 100, k).astype(int)],
            #    [(x,0) for x in np.random.uniform(0, 100, k).astype(int)],
            #    [(x,100) for x in np.random.uniform(0, 100, k).astype(int)],
            #    np.random.uniform(0, 100, size=(num_gen - 4*k - 4, 2)).astype(int)
            #
            
            #self.sites = np.vstack(site_arrays)
            self.sites = np.random.uniform(0, 100, size=(num_sites, 2)).astype(int)
            np.random.shuffle(self.sites)
               
            # Generate Edges
            delaunay = Delaunay(self.sites)
            self.simplices = delaunay.simplices
            
            # Remove some of the triangles
            self.sites = self.sites[:num_sites]
            sel0 = delaunay.simplices[:,0] < num_sites
            sel1 = delaunay.simplices[:,1] < num_sites
            sel2 = delaunay.simplices[:,2] < num_sites
            sel = sel0 & sel1 & sel2
            self.simplices = self.simplices[sel, :]
            
            # Build Neighbors Dictionary
            self.nhbrs = {x:set() for x in range(0,len(self.sites))}
            for row in self.simplices:
                for x in row:
                    temp = {y for y in row if y != x}
                    self.nhbrs[x] = self.nhbrs[x].union(temp)

            if self.check_solvable():
                break
        

        # History
        self.path = []
        self.distance = 0
        self.parent = None
        
        
        # Restore NumPy random state
        if random_state is not None:
            np.random.set_state(np_state)      


    def check_solvable(self):
        component = {self.state}
        new_sites = self.nhbrs[self.state]

        while True:
            if len(new_sites - component) == 0:
                return False
            if self.goal in new_sites:
                return True
            component = component.union(new_sites)
            nhbr_list = [self.nhbrs[s] for s in new_sites]
            
            new_sites = set().union(*nhbr_list)
            
        
    def __lt__(self, other):
        return self
    
    
    def copy(self):
        '''
        Return a copy of this instance. 
        '''
        new_node = AltRoutePlanning(self.num_sites, 2, is_copy=True)
        new_node.goal = self.goal
        new_node.sites = self.sites           # This is reference, not a copy. 
        new_node.simplices = self.simplices
        new_node.nhbrs = self.nhbrs
        new_node.path = []
        new_node.parent = self.parent
        new_node.state = self.state
        new_node.start = self.start
        new_node.distance = self.distance
        
        return new_node 
    
    
    def check_solved(self):
        return self.state == self.goal
    
    
    def get_actions(self):
        site = self.state
        return list(self.nhbrs[site])
    
    
    def take_action(self, a):
        new_node = self.copy()
        new_node.state = a
        new_node.parent = self
        
        s1 = self.sites[self.state]
        s2 = self.sites[new_node.state]
        new_node.distance += int(np.sum((s1 - s2)**2)**0.5)
        
        return new_node


    def random_walk(self, steps, random_state=None):
        if random_state is not None:
            np_state = np.random.get_state() 
            np.random.seed(random_state)
            
        new_node = self.copy()
        for i in range(steps):
            actions = new_node.get_actions()
            a = np.random.choice(actions)
            new_node = new_node.take_action(a)
            
        if random_state is not None:
            np.random.set_state(np_state)
            
        return new_node
    
    def get_path(self):
        
        if len(self.path) > 0:
            return self.path
        node = self
        while node.parent is not None:
            self.path.insert(0, node.state)
            node = node.parent
        
        return self.path
    
    
    def display(self, show_path=True, show_plot=True, figsize=[6,6], ps=100, window=None,
                labels=False, label_info={'offset':0, 'size':8}, border=False):
        '''
        Displays scatter plot of sites. 
        Option to display current path. 
        '''
        if show_plot == True:
            plt.figure(figsize=figsize)
        
        plt.triplot(self.sites[:,0], self.sites[:,1], self.simplices, c='#aaaaaa', linewidth=0.5)
        
        path = self.get_path()      # Get list of actions 
        path = [self.start] + path  # Extend to include starting site
        
        if show_path:   
            pts = self.sites[path, :]
            plt.plot(pts[:, 0], pts[:,1], zorder=1)
        
        start_x, start_y = self.sites[path[0],:]
        goal_x, goal_y = self.sites[self.goal,:]
        plt.scatter(start_x, start_y, c='green', s=ps, edgecolor='k', zorder=2)
        plt.scatter(goal_x, goal_y, c='red', s=ps, edgecolor='k', zorder=2)

        xlim = [0, 100]
        ylim = [0, 100]
        if window is not None:
            # window should be tuple of form (site, size)
            site, size = window
            c = self.sites[site,:]
            plt.xlim(c[0]-size, c[0]+size)
            plt.ylim(c[1]-size, c[1]+size)
            xlim = [c[0]-size, c[0]+size]
            ylim = [c[1]-size, c[1]+size]
        

        if labels:
            k = label_info['offset']
            fs = label_info['size']
            for i in range(self.num_sites):
                x, y = self.sites[i,:]
                if (x > xlim[0]) and (x < xlim[1]) and (y > ylim[0]) and (y < ylim[1]):
                    plt.text(self.sites[i,0]+k, self.sites[i,1]+k, s=i, fontsize=fs)
        
        if border == False:
            plt.axis('off')
            
        plt.xticks([])
        plt.yticks([])
        
        if show_plot:
            plt.show()
        
        
    def soln_info(self):
        path = self.get_path()
        return f'Path Length: {len(path)}, Path Cost: {self.path_cost()}'
        
        
    def get_state_id(self):
        return self.state
    
    
    def path_cost(self):
        return round(self.distance, 2)
           
       
    def heuristic(self, **kwargs):
        c = self.sites[self.state]
        g = self.sites[self.goal]
        d = np.sum((c-g)**2)**0.5
        return d.round(2)
        
        
if __name__ == '__main__':
    import os
    os.system('cls')
    state = AltRoutePlanning(num_sites=500, start=137, random_state=1)
    state = state.random_walk(steps=20, random_state=2)
    state.display(show_path=True)
    path = state.get_path()
    print(path)
    print(state.heuristic())
    
    