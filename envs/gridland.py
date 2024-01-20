import numpy as np
import matplotlib.pyplot as plt

class GridLand:
    
    def __init__(self, size, hills, random_state=None, is_copy=False):
        if size < 8:
            print('Warning: Size must be >= 8. Setting size to 8.')
            size = 8
        
        self.size = size
        self.hills = hills
        self.goal = (size-1, size-1)
        self.action_taken = None
        
        if is_copy: return 
        #-----------------------------------------------------------
        # Lines above executed for all nodes, copy or otherwise
        # Lines below are executed only for new nodes
        #-----------------------------------------------------------
        
        if random_state is not None:
            np_state = np.random.get_state()
            np.random.seed(random_state)
            
        self.gen_board()
        
        if random_state is not None:
            np_state = np.random.get_state()

        self.state = (0,0)
        self.path = []
        self.total_cost = self.board[0,0]
        self.parent = None
        self.root = self
    
        
    def gen_board(self):
        sz = self.size
        self.board = np.ones(shape=(3*sz, 3*sz)).astype(int)

        pos = np.random.choice(range(0, sz), size=(self.hills,2))
        # Add hills
        for i,j in pos:
            c = np.random.choice(range(1, sz//4))
            scale = np.random.uniform(0.5,4)
            hsz = 2 * c + 1
            dr = np.arange(hsz**2).reshape(hsz,hsz) // hsz
            d = ((dr - c)**2 + (dr.T - c)**2)**0.5
            h = np.ceil(scale * d).astype(int)
            h = h.max() - h
            r0, r1 = i + sz - c, i + sz + c + 1
            c0, c1 = j + sz - c, j + sz + c + 1
            self.board[r0:r1, c0:c1] += h
            
        self.board = self.board[sz:2*sz, sz:2*sz]
        self.board = self.board - self.board.min() + 1
    
    
    def check_solved(self):
        if self.state == self.goal:
            return True
        return False
        
    def __lt__(self, other):
        return self
    
    
    def copy(self):
        new_node = GridLand(self.size, self.hills, is_copy=True)
        new_node.board = self.board
        new_node.state = self.state
        #new_node.history = [*self.history]
        new_node.path = [*self.path]
        new_node.total_cost = self.total_cost
        new_node.parent = self.parent
        new_node.root = self.root
        new_node.action_taken = self.action_taken
        return new_node


    def display(self, show_fig=True):
        path = self.get_path()
        x = [0]
        y = [0]
        for a in path:
            if a == 0: x.append(x[-1]); y.append(y[-1]-1)
            if a == 1: x.append(x[-1]+1); y.append(y[-1])
            if a == 2: x.append(x[-1]); y.append(y[-1]+1)
            if a == 3: x.append(x[-1]-1); y.append(y[-1])
            
        plt.imshow(self.board, cmap='Blues')
        plt.plot(x,y, color='darkorange')
        
        ps = 10000 / self.size**2
        plt.scatter(self.goal[0], self.goal[1], marker="*", c='k', s=2*ps, zorder=2)
        plt.scatter(0, 0, marker="D", c='k', s=ps, zorder=2)
        plt.xticks([])
        plt.yticks([])
        if show_fig:
            plt.show()


    def soln_info(self):
        path = self.get_path()
        msg = f'Solution Length: {len(path)}, Total Cost: {self.total_cost}'
        return msg
        

    def get_actions(self):
        # 0: up, 1: right, 2: down, 3: left
        actions = []
        
        if self.state[0] != 0:  # Check if we can go up
            if self.board[self.state[0]-1, self.state[1]] != 0:
                actions.append(0)
        
        if self.state[1] != self.size-1:  # Check if we can go right
            if self.board[self.state[0], self.state[1]+1] != 0:
                actions.append(1)
        
        if self.state[0] != self.size-1:  # Check if we can go down
            if self.board[self.state[0]+1, self.state[1]] != 0:
                actions.append(2)  
        
        if self.state[1] != 0:  # Check if we can go left
            if self.board[self.state[0], self.state[1]-1] != 0:
                actions.append(3)

        return actions
    
    
    def take_action(self, a):
        if a not in self.get_actions():
            print(f'Action {a} is not valid in state {self.state}. This action will be ignored.')
            return self
        
        dr, dc = [(-1,0), (0,1), (1,0), (0,-1)][a]
        new_node = self.copy()
        new_node.state = (self.state[0] + dr, self.state[1] + dc)
        #new_node.history.append(a)
        #new_node.path.append(new_node.state)
        cost = self.board[new_node.state]
        new_node.total_cost += cost
        new_node.parent = self
        new_node.action_taken = a
        return new_node
    
    
    def get_path(self):
        import gc
        
        if len(self.path) > 0:
            return self.path
        
        node = self
        
        while node.parent is not None:
            self.path.insert(0, node.action_taken)
            node = node.parent
            
        gc.collect()
        return self.path
        
    
    def path_cost(self):
        return self.total_cost
    
    
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
    
    
    def get_state_id(self):
        return self.state
    
    
    def heuristic(self, alg='AST'):
        '''
        Wrapper function for heuristics. 
        '''
        if alg == 'AST': return self.ast_heuristic()
        if alg == 'GBF': return self.gbf_heuristic()


    def ast_heuristic(self, **kwargs):
        diff0 = self.goal[0] - self.state[0]
        diff1 = self.goal[1] - self.state[1]
        return diff0 + diff1
    

    def gbf_heuristic(self):
        i,j = self.state
        cost = self.board[i,j]
        
        diff0 = self.goal[0] - self.state[0]
        diff1 = self.goal[1] - self.state[1]
        
        return cost + diff0 + diff1



if __name__ == '__main__':
    import os
    os.system('cls')
    print('------------------------------------------')
    
    gw = GridLand(size=12, hills=12)
    print(gw.board)
    
    x = gw.random_walk(50)
    print(x.path)
    print(x.get_path())
    print(x.path)
    x.display()
    
        