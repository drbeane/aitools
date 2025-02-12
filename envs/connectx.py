import numpy as np
from IPython.display import clear_output
from time import sleep


class ConnectX:
    
    def __init__(self, rows=6, cols=7, goal=4, is_copy=False):
        self.rows = rows
        self.cols = cols
        self.goal = goal
        
        if is_copy:
            return
        
        self.cur_player = 1   # 1 or 2
        self.turns = 0
        self.winner = None
        self.game_over = False
        self.available_actions = list(range(1,self.cols + 1))
        
        # board is surrounded by a border of -1s
        self.board = np.zeros(shape=(rows+2, cols+2)).astype(int)
        self.board[0,:] = -1
        self.board[-1,:] = -1
        self.board[:,0] = -1
        self.board[:,-1] = -1

        # Counts the number of connections in any direction from cell
        # Player 1 connections are negative, Player 2 are positive
        # Also records if there is a free space at the end of the ray
        # Form of elts: (row#, col#, dir#, 0/1)  # 0 is a free space
        self.rays = np.zeros(shape=(rows+2, cols+2, 8, 2)).astype(int)
        self.rays[0,:,:,1] = 1
        self.rays[-1,:,:,1] = 1
        self.rays[:,0,:,1] = 1
        self.rays[:,-1,:,1] = 1
        
        self.rays[ 1, :, [5,6,7], 1] = 1  # Block down direction for bottom row
        self.rays[-2, :, [0,1,2], 1] = 1  # Block up direction for top row
        self.rays[ :, 1, [0,3,5], 1] = 1  # Block left direction for left col
        self.rays[ :,-2, [2,4,7], 1] = 1  # Block right direction for right col
        
        # Counts number of lines for each player with each possible number 
        # of freedoms at the end of the line (0, 1, or 2)
        # Form of elts: (player#, line_length, n_freedoms)
        self.line_counts = np.zeros(shape=(3, goal+1, 3)).astype(int)

        self.history = []


    def copy(self):
        new_state = ConnectX(self.rows, self.cols, self.goal, is_copy=True)
        new_state.cur_player = self.cur_player
        new_state.board = np.array(self.board)
        new_state.available_actions = [*self.available_actions]
        new_state.winner = self.winner
        new_state.turns = self.turns
        new_state.rays = np.array(self.rays)
        new_state.line_counts = np.array(self.line_counts)
        new_state.history = [*self.history]
        return new_state


    def display(self, raw=False, show_labels=True):
        '''
        Player 0 is X
        Player 1 is O
        '''
        if raw:
            print('------1--2--3--4--5--6--7----')
            print(self.board[::-1, ])
            print()
        else:
            c = {-1:'#', 0:'.', 1:'X', 2:'O'}
            b = self.board[1:-1, 1:-1][::-1, :]
            
            if self.cols < 10:
                labels = ' '
                for n in range(1, self.cols+1):
                    labels += f' {n}'
            else:
                lab1 = ' '
                lab2 = ' '
                for n in range(1, self.cols+1):
                    lab1 += '  ' if n < 10 else f' {str(n)[0]}'
                    lab2 += f' {n}' if n < 10 else f' {str(n)[1]}'
                    labels = lab1 + '\n' +  lab2
                
            
            if show_labels: print(labels)
            print('+' + '-'*(self.cols*2 + 1) + '+')
            
            for i in range(self.rows):
                print('|', end=' ')
                for j in range(self.cols):
                    print(c[b[i][j]], end=' ')
                    if j == self.cols-1:
                        print('|')
                    #    print(6 - i )
            print('+' + '-'*(self.cols*2 + 1) + '+')
        print()
    
    def replay_game(self, delay=1):
        print('Initial State')
        temp = ConnectX(self.rows, self.cols, self.goal)
        temp.display()
        for a in self.history:
            sleep(delay)
            clear_output(wait=True)
            print(f'Player {temp.cur_player} takes action {a}.')
            temp = temp.take_action(a)
            temp.display()
                    
    
    def get_actions(self):
        return self.available_actions

    
    def take_action(self, a):
        
        # Select column where play is made. 
        # Starting at the bottom (0) move up rows until finding an empty row. 
        column = self.board[:, a]
        r = 0
        while column[r] != 0: 
            r += 1
        
        # Copy board 
        new_state = self.copy()
        
        # Record the move
        new_state.board[r, a] = self.cur_player 
 
        # Remove the action if column is now full.
        if r == self.rows:
            new_state.available_actions.remove(a)
 
        # Update the rays
        new_state.update_rays(r,a)
        
        # Check if there is a winner or a tie
        new_state.check_for_win(r,a)
        
        # Increment the player and turns
        new_state.cur_player = self.cur_player % 2 + 1
        new_state.turns = self.turns + 1
        
        # Update move history
        new_state.history.append(a)
        
        # Return the new state
        return new_state

    
    def update_rays(self, r, c):
        current_player = self.cur_player
        other_player = (self.cur_player % 2) + 1

        rays = self.rays 

        # Loop over four cardinal directions.
        for d1 in range(4):
            d2 = 7 - d1  # Get opposite direction.

            # Get ray counts in both directions
            ray1 = rays[r,c,d1,0]
            ray2 = rays[r,c,d2,0]

            # Determine owner of each ray (p1 has neg counts, p2 has pos counts)
            owner1 = 1 if ray1 < 0 else 2 if ray1 > 0 else 0
            owner2 = 1 if ray2 < 0 else 2 if ray2 > 0 else 0

            own1 = (owner1 == self.cur_player)
            own2 = (owner2 == self.cur_player)

            # Calculate the number of connections including current piece in current direction 
            sign = (-1)**self.cur_player  # -1 for player1, 1 for player2)
            n_connected = ray1 * own1 + ray2 * own2 + sign
            
            # Determine updates for records at end of each ray
            update1 = [None, None]
            update2 = [None, None]
            
                        
            # Convert main dir to row and col deltas. Recall that board is flipped vertically.
            dr, dc = [(1, -1), (1, 0), (1, 1), (0, -1)][d1]
            
            block = (other_player in [owner1, owner2])
            
            # Update space at end of ray in main direction if it is open. 
            r1_ = r + dr * (1 + abs(rays[r,c,d1,0]))
            c1_ = c + dc * (1 + abs(rays[r,c,d1,0]))
            if self.board[r1_, c1_] == 0:
                # Determine new lengths
                update1[0] = n_connected if (owner1 in [0, self.cur_player]) else ray1
                # Block if rays owned by different players. Else, inherit from opposite direction
                update1[1] = 1 if block else rays[r,c,d2,1] 
                rays[r1_, c1_, d2, :] = update1
                        
            # Update space at end of ray in secondary direction if it is open.
            r2_ = r - dr * (1 + abs(rays[r,c,d2,0]))
            c2_ = c - dc * (1 + abs(rays[r,c,d2,0]))
            if self.board[r2_, c2_] == 0:
                # Determine new lengths
                update2[0] = n_connected if (owner2 in [0, self.cur_player]) else ray2
                # Block if rays owned by different players. Else, inherit from opposite direction
                update2[1] = 1 if block else rays[r,c,d1,1] 
                rays[r2_, c2_, d1, :] = update2

            #########################################
            # Update Line Counts
            #########################################
            
            ray1_blocked = rays[r,c,d1,1]           # 0 for no, 1 for yes
            ray2_blocked = rays[r,c,d2,1]           # 0 for no, 1 for yes

            # Update record for ray1 if it is non-trivial. 
            # If owned by current player, remove the ray. It will be merged with a larger one. 
            # If owned by other player, leave the ray but remove one of its freedoms. 
            if ray1 != 0:
                n_open1 = 2 - ray1_blocked
                self.line_counts[owner1, abs(ray1), n_open1] -= 1  # In either case, remove current record
                if owner1 != self.cur_player:
                    self.line_counts[owner1, abs(ray1), n_open1-1] += 1  # Add record with one less freedom

            # Update record for ray2 if it is non-trivial. 
            # If owned by current player, remove the ray. It will be merged with a larger one. 
            # If owned by other player, leave the ray but remove one of its freedoms. 
            if ray2 != 0:
                n_open2 = 2 - ray2_blocked
                self.line_counts[owner2, abs(ray2), n_open2] -= 1  # In either case, remove current record
                if owner2 != self.cur_player:
                    self.line_counts[owner2, abs(ray2), n_open2-1] += 1  # Add record with one less freedom

            # Determine new entry
            n_freedoms = (
                int((ray1_blocked == 0) * (owner1 != other_player) ) + 
                int((ray2_blocked == 0) * (owner2 != other_player) )
            )
            if abs(n_connected) > self.goal:
                n_connected = self.goal
            self.line_counts[self.cur_player, abs(n_connected), n_freedoms] += 1
                        
        # zero out rays for current position
        rays[r, c, :, :] = 0


    def ray_report(self):
        
        print('\nRay Report')
        print('Cell   Ray Lengths')
        print('----   -----------')
        for i in range(1,self.rows+1):
            for j in range(1, self.cols+2):
                if np.sum(self.rays[i,j,:,0] != 0) > 0:
                    print((i,j), self.rays[i,j,:,0])
        print()
        
        
    def check_for_win(self, row, col):
        ''' 
        Checks to see if the current player has won the game. 
        Returns their player number, if so. 
        Returns 0 if there is a tie. 
        '''
        if self.line_counts[self.cur_player, self.goal, :].sum() >= 1:
            self.game_over = True
            self.winner = self.cur_player
                    
        # Check for tie
        actions = self.get_actions()
        if len(actions) == 0:
            self.game_over = True
            self.winner = 0
        
        return None


    def heuristic(self, agent):
        ''' 
        Calculates a score for the state. 
        Score depends on whose turn it is, and the perspective.
        '''
        
        # Current player is the player whose turn it would be in this state. 
        # agent is the player who is evaluating the move. 
        cur_p = self.cur_player      # Current player
        next_p = cur_p % 2 + 1       # Next player
        lc = self.line_counts
                
        oppo = agent % 2 + 1
        score = 0
               
        agent_weight = 1.1 if self.cur_player == agent else 1
        oppo_weight = 1 if self.cur_player == agent else 1.1
        
        # Lines below make sense, but don't seem to improve performance. 
        #if self.line_counts[agent, self.goal, :].sum() > 0:
        #    return float('inf')
        #if self.line_counts[oppo, self.goal, :].sum() > 0:
        #    return float('-inf')
        
        # Loop over all line lengths
        for n in range(1, self.goal + 1):
            # Determine number of lines of current len for each player 
            agent_counts = self.line_counts[agent, n, :] * agent_weight
            oppo_counts = self.line_counts[oppo, n, :] * oppo_weight
            
            # Increment score. Multiplier at the end is for weighting based on length
            # We only care about lines with at least one freedom. 
            # We will double value of lines with 2 freedoms
            score += (agent_counts[1] - oppo_counts[1]) * 10**(n-1)
            score += 2 * (agent_counts[2] - oppo_counts[2]) * 10**(n-1)
            
        return round(score,1)
        

if __name__ == '__main__':
    
    state = ConnectX()
    

    
    