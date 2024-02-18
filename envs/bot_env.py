import numpy as np


class BotPlayerEnv:
    '''
    This is mean to represent a "stochastic" RL environment
    consisting of a 2P game environment and an agent-controlled opponent. 
    '''
    
    def __init__(self, game_env, agent, is_copy=False):
        if is_copy: return
            
        self.game_env = game_env.copy()
        self.agent = agent 
        
        '''
        self.state will likely need to be updated with each action taken. 
        Should this be replaced with a get_state() function?
        '''
        self.state = game_env.get_state()
        
        self.iteration = 1
        self.agent_player_num = 1
       
        self.terminal = False  # Required by TDAgent
    
        # History objects
        self.actions_taken = []
        self.path = [self.state]  # This will record only the states observed on player's turn. 
        self.rewards = []
    
    
    
    def copy(self):
        new_node = BotPlayerEnv(None, None, is_copy=True)
        new_node.game_env = self.game_env.copy() 
        new_node.state = self.state
        new_node.agent = self.agent
        new_node.iteration = self.iteration
        new_node.agent_player_num = self.agent_player_num
        new_node.terminal = self.terminal
        new_node.actions_taken = [*self.actions_taken]
        new_node.path = [*self.path]
        new_node.rewards = [*self.rewards]
        return new_node
    
    
    def reset(self, **kwargs):
        '''
        This is used with MC and TD methods at the start of each new episode. 
        '''
        
        new_node = self.copy()   
        new_node.iteration += 1
        new_node.agent_player_num = (new_node.agent_player_num % 2) + 1

        # If we are on an even iteration, then the bot gets the first move.         
        if new_node.iteration % 2 == 0:
            a_bot = self.agent.select_action(new_node.game_env)
            new_node.game_env = new_node.game_env.take_action(a_bot)
            new_node.state = new_node.game_env.get_state()

        new_node.path = [new_node.state]      
        
        return new_node
   
   
    def get_state(self):
        return self.game_env.get_state()
    
    
    def get_actions(self):
        return self.game_env.get_actions()
    
    
    def take_action(self, a, report=False):
        # Note: It will always be the RL player's turn when this is called. 
        player_num = (self.iteration - 1) % 2 + 1    # RL Player's number (1 or 2). 
        
        # Set reward structure
        reward_dict = {0:-50, 1:-100, 2:-100}
        reward_dict[player_num] = 100
        r = 0
        
        new_node = self.copy()
        s0 = self.get_state()
        
        # Need to check if action is valid. 
        # Actions are selected from a policy. We could easily get a 
        # policy that gives an illegal action. This is probably unavoidable 
        # since the argmax might select an illegal action from an untrained 
        # value function 
        if a not in self.get_actions():
            new_node.terminal = True
            new_node.game_env.winner = (player_num % 2) + 1
            #print('ILLEGAL MOVE!')

        else: 
            # Have player take action and get new state. 
            new_node.game_env = new_node.game_env.take_action(a)
            
        s1  = new_node.get_state()
    
        # Check if game is over. If so, return the reward.
        w = new_node.game_env.winner 
        if w is not None:
            new_node.terminal = True
            r = reward_dict[w]
            a_bot = None
        else:
            # If not, then the Bot (Agent) should make a move. 
            a_bot = self.agent.select_action(new_node.game_env)
            new_node.game_env = new_node.game_env.take_action(a_bot)
 
            # Check if game is over after the bot's action.
            w = new_node.game_env.winner 
            if w is not None:
                
                new_node.terminal = True
                r = reward_dict[w]

        new_node.rewards.append(r)
        
        if report:
            print(f'Player took action {a}.')
            if a_bot is not None:
                print(f'Bot took action {a_bot}.')
            if w is not None:
                if w == 0:
                    print('The game was a draw!')
                elif w == player_num:
                    print('The player won!')
                else:
                    print('The bot won!')
        
        return new_node

        

    