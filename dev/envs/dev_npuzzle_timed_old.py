#----------------------------------------------------------
# This version stores the entire path in each node.
# Copying the path when children are generated adds 
# significantly to the run time. 
# Also might have an impact on the memory requirements. 
#----------------------------------------------------------



import numpy as np
from IPython.display import clear_output
from time import sleep

class NPuzzle:
    
    def __init__(self, rows, cols, scramble=1000, random_state=None, 
                 is_copy=False, timer=None):
        ''' 
        Arguments:
            rows, cols -- dimensions of the board
            mode -- heuristic for greedy and a-star algs
                    "manhattan",                 
        '''
        self.rows = rows
        self.cols = cols
        self.N = rows * cols
        self.timer = timer
    
        if is_copy: return
        # Lines above executed for all nodes, copy or otherwise
        # Lines below are executed only for new nodes
        
        self.manhattan = 0
        self.conflicts = 0
        
        # History 
        self.path = []        
        
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
        self.timer.start('COPY')
        self.timer.start('COPY (New)')
        new_node = NPuzzle(self.rows, self.cols, is_copy=True)
        self.timer.stop('COPY (New)')
        self.timer.start('COPY (Board)')
        new_node.board = np.array(self.board)
        self.timer.stop('COPY (Board)')
        self.timer.start('COPY (solved_state)')
        new_node.solved_state = self.solved_state
        self.timer.stop('COPY (solved_state)')
        self.timer.start('COPY (blank)')
        new_node.blank = self.blank
        self.timer.stop('COPY (blank)')
        self.timer.start('COPY (path)')
        new_node.path = [*self.path]
        self.timer.stop('COPY (path)')
        self.timer.start('COPY (manhattan)')
        new_node.manhattan = self.manhattan
        self.timer.stop('COPY (manhattan)')
        self.timer.start('COPY (conflicts)')
        new_node.conflicts = self.conflicts
        self.timer.stop('COPY (conflicts)')
        self.timer.start('COPY (timer)')
        new_node.timer = self.timer
        self.timer.stop('COPY (timer)')
        self.timer.stop('COPY')
        
        return new_node
    
    
    def get_mem_req(self):
        import sys
        attributes = [
            self.rows, self.cols, self.N, self.manhattan, self.conflicts,
            self.path, self.board, self.solved_state, self.blank
        ]
        mem = sum([sys.getsizeof(a) for a in attributes])
        return mem
    
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
        self.timer.start('TAKE ACTION')
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
        self.timer.start('TAKE ACTION (Get Target Value)')
        target_val = self.board[target_row, target_col]
        new_node = self.copy()
        self.timer.stop('TAKE ACTION (Get Target Value)')
        
        new_node.board[target_row, target_col] = 0
        new_node.board[blank_row, blank_col] = target_val
        new_node.blank = (target_row, target_col)

        # Update Manhattan Heuristic
        # We need to look at metric for old and new positions for tile being moved.
        self.timer.start('TAKE ACTION (Update Manhattan)')
        correct_row, correct_col = (target_val - 1)//self.cols, (target_val - 1)%self.cols
        old_diff = abs(correct_row - target_row) + abs(correct_col - target_col) 
        new_diff = abs(correct_row - blank_row) + abs(correct_col - blank_col)
        new_node.manhattan = new_node.manhattan - old_diff + new_diff
        self.timer.stop('TAKE ACTION (Update Manhattan)')
        
        # Update Linear Conflicts: [Case 1] Tile moving up or down
        self.timer.start('TAKE ACTION (Update Linear Conflicts)')
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
        self.timer.stop('TAKE ACTION (Update Linear Conflicts)')
        
        # Record the action 
        new_node.path.append(a)
                
        self.timer.stop('TAKE ACTION')
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
        
    
    def soln_info(self):
        msg = f'Solution Length: {len(self.path)}'
        return msg 

    def animate_solution(self, path, delay=1):
        print('Initial State')
        self.display()
        temp = self.copy()
        for a in path:
            sleep(delay)
            clear_output(wait=True)
            temp = temp.take_action(a)
            print(f'Action {a} was taken.')
            temp.display()    
        

    def check_solved(self):
        '''
        Checks if the puzzle has been solved. 
        '''
        return np.all(self.board == self.solved_state)


    def get_state_id(self):
        return str(self.board.tolist())
    
    
    def path_cost(self):
        return len(self.path)
    
    
    def heuristic(self, **kwargs):
        return self.manhattan + 2 * self.conflicts
    

if __name__ == '__main__':
    
    puzzle = NPuzzle(4, 4, scramble=400)
    puzzle.display()
    