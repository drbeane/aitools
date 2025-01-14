class NPuzzle:
    
    def __init__(self, rows, cols, scramble=1000, random_state=None, is_copy=False):
        ''' 
        Arguments:
            rows, cols -- dimensions of the board
            mode -- heuristic for greedy and a-star algs
                    "manhattan",                 
        '''
        import numpy as np
        
        self.rows = rows
        self.cols = cols
        self.N = rows * cols
        self.action_taken = None
        
        if is_copy: return
        #-----------------------------------------------------------
        # Lines above executed for all nodes, copy or otherwise
        # Lines below are executed only for new nodes
        #-----------------------------------------------------------
        
        self.manhattan = 0
        self.conflicts = 0
        
        # History 
        self.path = []
        self.action_count = 0
        self.parent = None
        self.root = self
        
        # Configure Board
        self.board = np.arange(1,self.N+1).reshape(rows, cols)
        self.board[-1, -1] = 0
        self.solved_state = np.array(self.board)
        self.blank = (rows-1, cols-1)
        if scramble > 0:
            self.scramble(num_moves=scramble, random_state=random_state)
        

    def __lt__(self, other):
        return self
    
    
    def copy(self):
        '''
        Returns a copy of this instance. 
        '''
        import numpy as np
        
        new_node = NPuzzle(self.rows, self.cols, is_copy=True)
        new_node.board = np.array(self.board)
        new_node.solved_state = self.solved_state
        new_node.blank = self.blank
        new_node.path = [*self.path]
        new_node.manhattan = self.manhattan
        new_node.conflicts = self.conflicts
        new_node.parent = self.parent
        new_node.action_count = self.action_count
        new_node.action_taken = self.action_taken
        new_node.root = self.root
        
        return new_node
    
    
    def get_actions(self):
        '''
        Returns a list of valid actions based on location of blank tile. 
        '''
        actions = []       
        row, col = self.blank
        
        # 0:up, 1:left, 2:down, 3:right
        if row != 0: actions.append(0)
        if col != self.cols - 1: actions.append(1)
        if row != self.rows - 1: actions.append(2)
        if col != 0: actions.append(3)

        return actions
    
    
    def take_action(self, a):
        '''
        Applies the selected action, if valid. Assumes that the action is valid. 
        '''
        # Actions to Directions: 0-up, 2-down, 1-right, 3-left
        # These indicate the direction the blank cell will be moved. 
        # Get location of blank cell and location of target cell
        blank_row, blank_col = self.blank
        if a == 0: target_row, target_col = (blank_row - 1, blank_col)
        if a == 2: target_row, target_col  = (blank_row + 1, blank_col)
        if a == 3: target_row, target_col  = (blank_row, blank_col - 1)
        if a == 1: target_row, target_col  = (blank_row, blank_col + 1)
        
        # Get target value. 
        target_val = self.board[target_row, target_col]
        new_node = self.copy()
        
        new_node.board[target_row, target_col] = 0
        new_node.board[blank_row, blank_col] = target_val
        new_node.blank = (target_row, target_col)

        # Update Manhattan Heuristic
        # We need to look at metric for old and new positions for tile being moved.
        correct_row, correct_col = (target_val - 1)//self.cols, (target_val - 1)%self.cols
        old_diff = abs(correct_row - target_row) + abs(correct_col - target_col) 
        new_diff = abs(correct_row - blank_row) + abs(correct_col - blank_col)
        new_node.manhattan = new_node.manhattan - old_diff + new_diff
        
        # Update Linear Conflicts: [Case 1] Tile moving up or down
        if a in {0,2}:
            if target_row == correct_row:    # Target is leaving correct row
                # This could reduce conflicts. Count conflicts existing BEFORE move.
                new_node.conflicts -= self.count_row_conflicts(target_row, target_col)
            if blank_row == correct_row:      # Target is entering correct row
                # This could increase conflicts. Count conflicts existing AFTER move.
                new_node.conflicts += new_node.count_row_conflicts(blank_row, blank_col)
        
        # Update Linear Conflicts: [Case 1] Tile moving left or right
        if a in {1,3}:
            if target_col == correct_col:     # Target is leaving correct column
                # This could reduce conflicts. Count conflicts existing BEFORE move.
                new_node.conflicts -= self.count_col_conflicts(target_row, target_col)
            if blank_col == correct_col:      # Target is entering correct column
                # This could increase conflicts. Count conflicts existing AFTER move.
                new_node.conflicts += new_node.count_col_conflicts(target_row, blank_col)
        
        # Record the action 
        new_node.path.append(a)
        new_node.parent = self
        new_node.action_count = new_node.action_count + 1
        new_node.action_taken = a
            
        return new_node


    def count_row_conflicts(self, row, col):
        '''
        Returns number of row-based linear conflicts tile in (row, col) contributes to.
        Used in take_action() to update conflict count when a move is made. 
        '''
        val = self.board[row, col]
        if val == 0:
            return 0
        if (val - 1) // self.cols != row:
            return 0
        
        
        row_vals = self.board[row, :]
        
        n_conflicts = 0
        mode = 1
        for x in row_vals:
            if mode == 1:
                if x > val:
                    if (x - 1) // self.cols == row:
                        n_conflicts += 1
                elif x == val:
                    mode = 2
            else:
                if x < val:
                    if (x - 1) // self.cols == row:
                        if x > 0:
                            n_conflicts += 1
        
        return n_conflicts
                    
    
    def count_col_conflicts(self, row, col):
        '''
        Returns number of col-based linear conflicts tile in (row, col) contributes to.
        Used in take_action() to update conflict count when a move is made. 
        '''
        val = self.board[row, col]
        if val == 0:
            return 0
        if (val - 1) % self.cols != col:
            return 0
        
        
        col_vals = self.board[:, col]
        
        n_conflicts = 0
        mode = 1
        for x in col_vals:
            if mode == 1:
                if x > val:
                    if (x - 1) % self.cols == col:
                        n_conflicts += 1
                elif x == val:
                    mode = 2
            else:
                if x < val:
                    if (x - 1) % self.cols == col:
                        if x > 0:
                            n_conflicts += 1
        
        return n_conflicts


    def scramble(self, num_moves=1000, random_state=None):
        '''
        Scrambles the puzzle. 
        '''
        import numpy as np
        
        if random_state is not None:
            np_state = np.random.get_state()
            np.random.seed(random_state)
        
        new_state = self.copy()
        for i in range(num_moves):
            a = np.random.choice(new_state.get_actions())
            new_state = new_state.take_action(a)
        
        self.board = np.array(new_state.board)
        self.blank = new_state.blank  
        self.manhattan = new_state.manhattan
        self.conflicts = new_state.conflicts

        if random_state is not None:
            np.random.set_state(np_state)


    def display(self):
        '''
        Displays the puzzle. 
        '''
        k = len(str(self.N))
        print('+' + '-'*(k+1)*self.cols + '-+')
        for row in self.board:
            print('| ', end='')
            for x in row:
                c = '-' if x == 0 else x
                print(f'{c:>{k}}', end=' ')
            print('|')
        print('+' + '-'*(k+1)*self.cols + '-+')
        
    
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
    
    
    def soln_info(self):
        msg = f'Solution Length: {self.action_count}'
        return msg 

    def animate_solution(self, path=None, delay=1):
        from IPython.display import clear_output
        from time import sleep
        
        path = self.get_path()
        print('Initial State')
        node = self.root.copy()
        node.display()
        for a in path:
            sleep(delay)
            clear_output(wait=True)
            node = node.take_action(a)
            print(f'Action {a} was taken.')
            node.display()    
        

    def check_solved(self):
        '''
        Checks if the puzzle has been solved. 
        '''
        import numpy as np
        
        return np.all(self.board == self.solved_state)


    def get_state_id(self):
        '''
        Returns identifier for current state. 
        State id is a string recording board state.
        '''
        return str(self.board.tolist())
    
    
    def path_cost(self):
        return self.action_count
    
    
    def heuristic(self, **kwargs):
        return self.manhattan + 2 * self.conflicts
    

    def generate_image(self, show=False):
        import cv2
        import matplotlib.pyplot as plt
        import numpy as np
        
        rows, cols = self.rows, self.cols
        box_size = 64
        image = np.zeros((rows*box_size, cols*box_size, 3))#.astype(int)

        for r in range(rows):
            for c in range(cols):
                y0, x0, y1, x1 = r*box_size, c*box_size, (r+1)*box_size, (c+1)*box_size
                v = self.board[r,c]
                if v == 0:
                    image = cv2.rectangle(image, (x0, y0), (x1, y1), color=(1, 1, 1), thickness=-1) 
                else:
                    image = cv2.rectangle(image, (x0, y0), (x1, y1), color=(32,32,32), thickness=-1) 
                    x_margin = 18 if v < 10 else 2
                    y_margin = 48
                    image = cv2.putText(image, str(v), (x0 + x_margin, y0 + y_margin), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255,255,255), 3) 

                image = cv2.rectangle(image, (x0+1, y0+1), (x1-2, y1-2), color=(255, 255, 255), thickness=2) 
    
        image = (255 * image).astype(np.uint8)

        if show:
            plt.figure(figsize=(3,3))
            plt.imshow(image, cmap='bone_r')
            plt.grid()
            plt.axis('off')
            plt.show()

        return image

    def generate_gif(self, fps=1):
        import imageio
        import os
        from IPython.display import Image, display

        frames = [self.generate_image()] * 2 * fps

        node = self
        while node.parent is not None:
            node = node.parent
            f = node.generate_image()
            frames.insert(0, f)

        for i in range(fps):
            frames.insert(0, f)

        os.makedirs('gifs', exist_ok=True)
        n = len(os.listdir('gifs/')) + 1
        filename = f'gifs/soln_gif_{n}.gif'
        imageio.mimsave(filename, frames, format='GIF', duration=1000/fps, loop=0)   

        with open(filename,'rb') as f:
            display(Image(data=f.read(), format='png'))



if __name__ == '__main__':
    
    puzzle = NPuzzle(4, 4, scramble=400)
    puzzle.display()
    