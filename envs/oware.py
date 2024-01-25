import numpy as np
from IPython.display import clear_output
from time import sleep

class Oware:
    
    def __init__(self, ppp=6, spp=4, is_copy=False):
        '''
        Constructor. 
        ppp - Pits per player
        spp - Seeds per pit
        '''
        self.ppp = ppp
        self.spp = spp
        self.pits = 2 * self.ppp    # Total number of pits
        
        if is_copy: return 
        
        self.board = self.spp * np.ones(self.pits).astype(int)
        self.winner = None
        self.turns = 0
        self.cur_player = 1 # 1 or 2
        self.score = {1:0, 2:0}
        self.history = []
        self.action_taken = None
        self.parent = None
        self.root = self
    
    
    def copy(self):
        new_state = Oware(is_copy=True)
        new_state.board = np.array(self.board)
        new_state.winner = self.winner
        new_state.turns = self.turns
        new_state.cur_player = self.cur_player
        new_state.score = self.score.copy()
        new_state.history = [*self.history]
        new_state.action_taken = self.action_taken
        new_state.parent = self.parent

        new_state.root = self if self.parent is None else self.root
        
        return new_state
    
    
    def display(self):
        print('+' + '---' * self.ppp + '-+\n|', end='')    
        for x in self.board[:self.ppp]:
            print(f'{x:>3}', end='')
        print(f' | Player 1: {self.score[1]} points\n|', end='')
        for x in np.flip(self.board[self.ppp:]):
            print(f'{x:>3}', end='')
        print(f' | Player 2: {self.score[2]} points')
        print('+' + '---' * self.ppp + '-+\n')    
    
    
    def get_history(self):
        if len(self.history) > 0:
            return self.history
        
        node = self.parent
        while node.parent is not None:
            self.history.insert(0, node.action_taken)
            node = node.parent
        
        return self.history 
    
    def replay_game(self, delay=1):
        history = self.get_history()
        print('Initial State')
        temp = Oware(self.ppp, self.spp)
        temp.display()
        for a in history:
            sleep(delay)
            clear_output(wait=True)
            print(f'Player {temp.cur_player} takes action {a}.')
            temp = temp.take_action(a)
            temp.display()
    
    def generate_image(self, shade_action=False, show=False):
        import cv2
        import matplotlib.pyplot as plt
        import numpy as np
        
        box_size=64
        border = 2
        image = np.zeros((2*box_size + 2*border, 6*box_size + 2*border, 3))
        
        for r in range(2):
            for c in range(6):
                y0 = r*box_size + 2*border
                y1 = (r+1)*box_size
                x0 = c*box_size + 2*border
                x1 = (c+1)*box_size
                i = c if r == 0 else -c-1
                v = self.board[i]
                
                color = (240, 240, 240)
                if shade_action: 
                    
                    a = self.action_taken
                    if a == 6*r + c:
                        color = (200, 255, 255) if a < 6 else (255, 200, 255)
                
                image = cv2.rectangle(image, (x0+1, y0+1), (x1-2, y1-2), color=color, thickness=-1) 
                x_margin = 16 if v < 10 else 0
                y_margin = 44
                image = cv2.putText(image, str(v), (x0 + x_margin, y0 + y_margin), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0,0,0), 3) 
    
        image = image.astype(np.uint8)

        if show:
            plt.figure(figsize=(6,6))
            plt.imshow(image)
            #plt.imshow(image, cmap='bone_r')
            plt.grid()
            #plt.axis('off')
            plt.show()

        return image
                
    def generate_gif(self, fps=1):
        import imageio
        import os
        from IPython.display import Image, display

        frames = [self.generate_image()] * 2 * fps
        
        node = self.parent
        while node is not None:
            f = node.generate_image(shade_action=True) 
            frames.insert(0, f)
            f = node.generate_image(shade_action=False) 
            frames.insert(0, f)
            node = node.parent
        
        for i in range(fps):
            frames.insert(0, f)


        os.makedirs('gifs', exist_ok=True)
        n = len(os.listdir('gifs/')) + 1
        filename = f'gifs/soln_gif_{n}.gif'
        imageio.mimsave(filename, frames, format='GIF', duration=1000/fps, loop=0)   
        
        with open(filename,'rb') as f:
            display(Image(data=f.read(), format='png'))
        
        
    def get_actions(self):
        start = (self.cur_player - 1) * self.ppp               # Index of 1st pit for cur player
        pits = self.board[start:start + self.ppp]              # Seed count for cur player's pits
        op_start = (self.cur_player % 2) * self.ppp            # Index of 1st pit for opponent
        oppo_pits = self.board[op_start:op_start + self.ppp]   # Seed counts for opponent. 
        
        # If opponent has seeds, current player can select any pit they own with seeds
        # Otherwise, cur_player must select a move that gives seeds to opponent. 
        # Such a move will always be possible when this method is called. 
        # The check_for_win() method will end the game if this is not possible. 
        if oppo_pits.sum() > 0:
            actions = np.argwhere(pits != 0).reshape(-1,)
        else:
            sel = (pits + np.arange(self.ppp)) >= self.ppp
            actions = np.argwhere(sel).reshape(-1,)
            
        return actions + start
    
    
    def take_action(self, a):
        new_state = self.copy()
        
        seeds = new_state.board[a]
        new_state.board[a] = 0
        k = new_state.pits - 1
        seed_dist = np.ones(k).astype(int) * seeds // k
        seed_dist[:(seeds % k)] += 1
        
        after = seed_dist[:(k-a)]
        before = seed_dist[(k-a):]
        
        new_state.board[(a+1):] += after
        new_state.board[:a] += before
        
        last_pit = (a + (seeds % k)) % new_state.pits
        
        # Check to see if last pit played in is on opponent side
        if (last_pit // self.ppp) + 1 != new_state.cur_player:
            
            start = (new_state.cur_player % 2) * self.ppp
            oppo_pits = new_state.board[start:start + self.ppp]
            
            # Check to see if there is a capture
            if new_state.board[last_pit] in [2,3]:
                # Determine the starting pit for the capture
                i = last_pit - 1
                start_capture = last_pit
                while i >= start:
                    if new_state.board[i] in [2,3]:
                        start_capture = i
                    else:
                        break
                    i -= 1
                    
                seeds_captured = new_state.board[start_capture: last_pit + 1].sum()
                
                # See if move would capture all of opponent's seeds.
                # If so, then NO seeds are captured. 
                if seeds_captured != oppo_pits.sum():
                    new_state.board[start_capture: last_pit + 1] = 0
                    new_state.score[new_state.cur_player] += seeds_captured
        
        #new_state.history.append(a)
        new_state.parent = self
        self.action_taken = a
        
        # Increment number of turns and the current player. 
        new_state.turns += 1
        new_state.cur_player = (new_state.cur_player % 2) + 1
        
        # Check to see if a player has won. 
        new_state.check_for_win()
        
        return new_state
    
    
    def check_for_win(self):
        
        # Check to see if there are no legal moves for the next player. 
        # This should only happen if the previous player cleared their board
        # and the current player is unable to give the prev player seeds.
        # In this case, the current player captures all remaining seeds
        if len(self.get_actions()) == 0:
            self.score[self.cur_player] += self.board.sum()
            self.board = self.board * 0

        # Check for two-seed edge case that results in infinite game
        if self.board.sum() == 2:
            if self.board[0] == 1 and self.board[self.ppp] == 1:
                self.score[1] += 1
                self.score[2] += 1
                self.board = self.board * 0
            
        # Check to see if either player has over half the seeds.
        if self.score[1] > (self.pits * self.spp) / 2:
            self.winner = 1
        if self.score[2] > (self.pits * self.spp) / 2:
            self.winner = 2
        
        # Check to see if there is a tie. 
        if self.score[1] == (self.pits * self.spp) / 2:
            if self.score[2] == (self.pits * self.spp) / 2:
                self.winner = 0
        
        return None
        
    
    def heuristic(self, agent, **kwargs):
        oppo = (agent % 2) + 1
        return  self.score[agent] - self.score[oppo]
        