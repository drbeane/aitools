################################################################
# Notes: 1/12/2023
# 1. MinMax agent has an unused "num_random_turns" parameter
# 2. Consider adding a HumanPlayer agent. 
################################################################

import numpy as np
import time as time
from tqdm import tqdm


def play_game(root, agents, display_flags='', max_turns=None, 
              return_times=False, random_state=None):
    '''
    Plays a single game with 2 agents selecting actions. 
    
    root - Root node for game environment.
    agents - List of 2 agents, one for each player.
    
    Display flags: 
        a - Show each action taken.
        s - Show game state after each action. 
        w - State the winner of the game.
        t - Report time taken by each player.
    '''
    
    # Set the random seed, if one is specified. 
    if random_state is not None:
        np_state = np.random.get_state()
        np.random.seed(random_state)
    
    agents_dict = {1:agents[0], 2:agents[1]}   # Maps player number to agent. 
    play_time = {1:0, 2:0}                     # Tracks play time for each agent. 
    
    game_state = root.copy()                   # Create a copy of the game env
    
    if 's' in display_flags:
        game_state.display()
    
    while game_state.winner is None:           # Loop until there is a winner (or a tie)
        print('------ STARTING LOOP')
        print(f'1. {game_state.turns=}')
        cp = game_state.cur_player             # Determine current player
        print(f'2. {game_state.turns=}')
        agent = agents_dict[cp]                # Lookup agent for current player
        
        t0 = time.time()                
        print(f'3. {game_state.turns=}')    
        a = agent.select_action(game_state)    # Agent selects an action. 
        play_time[cp] += time.time() - t0 
        print(f'4. {game_state.turns=}')    
        
        if a == 'quit':
            print('Exiting game.')
            return
        print(f'{game_state.turns=}')
        print(f'{a=}')
        game_state = game_state.take_action(a) # Apply the selected action. 
        
        # Report the action taken, if requested. 
        if 'a' in display_flags:
            print(f'Turn {game_state.turns}: Player {cp} ({agent.name}) takes Action {a}')
        
        # Display new game state, if requested. 
        if 's' in display_flags:
            game_state.display()
        
        # Display the winnder, if needed. 
        if ('w' in display_flags) and (game_state.winner is not None):
            #if game_state.winner is None:
            #    pass
            if game_state.winner > 0:
                winner = game_state.winner
                winner_name = agents_dict[winner].name
                print(f'Player {winner} ({winner_name}) wins!')
            elif game_state.winner == 0:
                print('\nThere is a tie!')
        

        # End the loop if the number of turns has reach the maximum. 
        if game_state.turns == max_turns:
            break
    
    # Display play time report, if requested.
    if 't' in display_flags:
        for i in [1,2]:
            print(f'Player {i} took {play_time[i]:.4f} seconds.')
    
    # Reset the numpy random state
    if random_state is not None:
        np.random.set_state(np_state)
    
    # The tournament function below needs access to play_time list. 
    if return_times:
        return game_state, play_time
    else:
        return game_state
        

def tournament(root, agents, rounds, switch_players=True, reroll=False, random_state=None,
               display_results=True, return_results=False, show_progress=True):
    '''
    Runs a tournament between two agents. 
    
    root - Root node for game environment.
    agents - List of 2 agents, one for each player.
    rounds - Number of games to play. 
    switch_players - Agents will alternate 1st player if True. 
    reroll - Allows game envs with stochastic setups to be rerolled after each game. 
    display_results - Print a report of the tournament results. 
    return_results - 
    random_state - Random seed used for tournament. 
    show_progress - Determines if a progress bar should be displayed.
    '''
    
    # Set the random seed, if one is specified. 
    if random_state is not None:
        np_state = np.random.get_state()
        np.random.seed(random_state)
    
    win_counts = [0, 0, 0]          # Tie, Agent 1, Agent 2
    total_play_time = {1:0, 2:0}    # Records total play time for each player. 
    turn_count = 0                  # Total number of turns taken. Used to report avg turns. 
    
    # Determine if a progress bar should be displayed. 
    rng = range(rounds)
    if display_results and show_progress: 
        rng = tqdm(range(rounds))
            
    # Iterate over rng, playing one game for each iteration. 
    for i in rng:
        
        # If players are alternating, we need to update the player-to-agent map
        # Note that this list is also used below to update the win counts. 
        player_to_agent = [0, 1, 2]
        if switch_players:
            player_to_agent = [0, 1 + i%2, 2 - i%2]
        
        # Assign agents to players for current game. 
        p1 = agents[i % 2]
        p2 = agents[1 - i%2]
        
        # We need to reset the game to an initial state. 
        # For games with a stochastic setup, we should reroll the initial state. 
        if reroll:
            init_state = root.reroll()
        else:
            init_state = root.copy()
        
        # Play the game.
        final_state, play_time = play_game(init_state, [p1, p2], return_times=True)

        # Update win counts for each agent. We need to map player to agent. 
        win_counts[player_to_agent[final_state.winner]] += 1
        
        # Update turn count and play times. 
        turn_count += final_state.turns    
        for i in [1,2]:
            a = player_to_agent[i]
            total_play_time[a] += play_time[i]
    
    # Display the results of the tournament, if request. 
    if display_results:
        n0 = agents[0].name
        n1 = agents[1].name
        
        d = abs(len(n0) - len(n1))
        m = max(len(n0), len(n1))
        b = max(0, 17 - m)
        
        ex_sp0 = b if len(n0) == max([len(n0), len(n1)]) else b+d
        ex_sp1 = b if len(n1) == max([len(n0), len(n1)]) else b+d
        t_sp = max(19, m+2)
        avg_sp = max(0, m-17)
        
        title = f'{n0} vs. {n1}'
        print(title)
        print('-' * len(title))
        print(f'Ties: {" "*t_sp}{win_counts[0]}')
        print(f'{n0} Wins:  {" "*ex_sp0}{win_counts[1]}')
        print(f'{n1} Wins:  {" "*ex_sp1}{win_counts[2]}')
        print(f'{n0} took:  {" "*ex_sp0}{total_play_time[1]:.2f} seconds')
        print(f'{n1} took:  {" "*ex_sp1}{total_play_time[2]:.2f} seconds')
        print(f'Average number of turns: {" "*avg_sp}{turn_count/rounds:.1f}')

    # Reset the numpy random state
    if random_state is not None:
        np.random.set_state(np_state)

    # Return the results, if requested. 
    # This is used only for HW 3, Part 4. 
    if return_results:
        return {
            'win_counts':win_counts, 
            'play_time':total_play_time, 
            'avg_turns':round(turn_count/rounds,1)
        }
        
   
class RandomPlayer:
    '''
    Implements a Random Player agent. 
    Agent selects actions at random from those available. 
    '''
    def __init__(self, name):
        self.name = name
        
    def select_action(self, state):
        actions = state.get_actions()       
        i = np.random.choice(len(actions))
        return actions[i]        


class GreedyPlayer:
    '''
    Implements a Greedy Player agent. 
    Agent selects actions greedily according to some heuristic. 
    '''

    def __init__(self, name):
        self.name = name
    
    def select_action(self, state):
        # Get available actions. 
        actions = state.get_actions()
        actions = np.array(actions)
        
        # Get new states
        states = [state.take_action(a) for a in actions]
        
        # Score each state. 
        cp = state.cur_player
        scores = [s.heuristic(agent=cp) for s in states]
        scores = np.array(scores)
        
        # Find states with best scores
        max_score = scores.max()
        sel = (scores == max_score)
        best_actions = actions[sel]
        
        i = np.random.choice(len(best_actions))
        return best_actions[i]              
    
    
class MinimaxPlayer:
    
    def __init__(self, name, depth, ABP=True, random_turns=0):
        self.name = name
        self.depth = depth    
        self.num_rand = random_turns
        self.ABP = ABP
    
    def select_action(self, state):
        # Record the player number for agent making the move. 
        self.agent = state.cur_player
        
        # Create alpha and beta for AB Pruning
        alpha = float('-inf') 
        beta = float('inf') 
        
        best_score = float('-inf')
        best_actions = []
        
        # Loop over all possible actions/children
        for a in state.get_actions():
            
            # Evaluate child, which is in a Min layer. 
            child_state = state.take_action(a)
            v = self.min_valuation(child_state, alpha, beta, self.depth - 1)

            # Update alpha
            alpha = max(alpha, v)
            
            # Check to see if child score is better than current best
            if v == best_score:
                best_actions.append(a)
            if v > best_score: 
                best_score = v   
                best_actions = [a]

        i = np.random.choice(len(best_actions))
        return best_actions[i]              

        
    def min_valuation(self, state, alpha, beta, depth):
        # The current player for states in a min layer is not the agent. 
        # But the states are evaluated from the persective of the agent. 
        
        # Check to see if the current state is terminal. 
        if state.winner is None:
            pass
        elif state.winner == 0:
            return 0                        # Tied state
        elif state.winner == self.agent:
            return float('inf')             # Agent wins in this state
        else:
            return float('-inf')            # Agent loses in this state
        # The else condition above should never be reached. 
        # Since we are in a min layer, the agent would be the last player to
        # have taken an action. That can't result in opponent win.  
        
        # If the state is not terminal, we continue with valuation. 
        
        # If depth == 0, then state is scored according to the heuristic. 
        if depth == 0:
            # Note that agent will always be different from cur_player in this fn call
            return state.heuristic(agent=self.agent)
         
        # If depth > 0, then state is evaluated as min of child states. 
        # Child states will be in a max layer. 
        
        min_v = float('inf')
        for a in state.get_actions():
            # Evaluate child, which is in a Max layer. 
            child_state = state.take_action(a)
            v = self.max_valuation(child_state, alpha, beta, depth - 1)
            min_v = min(v, min_v)
            
            # Update beta
            beta = min(beta, min_v)
            
            if self.ABP and min_v < alpha:
                return min_v
            
        return min_v
        
    def max_valuation(self, state, alpha, beta, depth):
        # The current player for states in a max layer is the agent. 
        # States are evaluated from the persective of the agent. 
        
        # Check to see if the current state is terminal. 
        if state.winner is None:
            pass
        elif state.winner == 0:
            return 0                        # Tied state
        elif state.winner == self.agent:
            return float('inf')             # Agent wins in this state
        else:
            return float('-inf')            # Agent loses in this state
        # The 2nd to last condition above should never be reached. 
        # Since we are in a max layer, oppo would be the last player to
        # have taken an action. That can't result in agent win.  
        
        # If the state is not terminal, we continue with valuation. 
        
        # If depth == 0, then state is scored according to the heuristic. 
        if depth == 0:
            # Note that agent will always be cur_player in this fn call
            return state.heuristic(agent=self.agent)
         
        # If depth > 0, then state is evaluated as max of child states. 
        # Child states will be in a min layer. 
        max_v = float('-inf')
        for a in state.get_actions():
            # Evaluate child, which is in a Min layer. 
            child_state = state.take_action(a)
            v = self.min_valuation(child_state, alpha, beta, depth - 1)
            max_v = max(v, max_v)

            # Update alpha            
            alpha = max(alpha, max_v)
            
            if self.ABP and max_v > beta:
                return max_v
            
        return max_v
  
    
             
class MinimaxPlayerABP_RETIRED:
    
    def __init__(self, name, depth, random_turns=0):
        self.name = name
        self.depth = depth    
        self.num_rand = random_turns
        
    def select_action(self, state):
        # Obtain the player number for agent making the move. 
        self.agent = state.cur_player

        alpha = float('-inf')
        beta = float('inf')
        best_score = float('-inf')
        best_actions = []
        
        # Loop over all available actions/children.
        for a in state.get_actions():
            
            # Evaluate child, which is in a Min layer. 
            child_state = state.take_action(a)
            v = self.min_valuation(child_state, alpha, beta, self.depth - 1)
            
            # Update alpha
            alpha = max(alpha, v)
            
            # Check to see if child score is better than current best
            if v == best_score:
                best_actions.append(a)
            if v > best_score: 
                best_score = v   
                best_actions = [a]
        
        i = np.random.choice(len(best_actions))
        return best_actions[i]

    def min_valuation(self, state, alpha, beta, depth):
        # The current player for states in a min layer is not the agent. 
        # But the states are evaluated from the persective of the agent. 
        
        # Check to see if the current state is terminal. 
        if state.winner is None:
            pass
        elif state.winner == 0:
            return 0                        # Tied state
        elif state.winner == self.agent:
            return float('inf')             # Agent wins in this state
        else:
            return float('-inf')            # Agent loses in this state
        # The else condition above should never be reached. 
        # Since we are in a min layer, the agent would be the last player to
        # have taken an action. That can't result in opponent win.  
        
        # If the state is not terminal, we continue with valuation. 
        
        # If depth == 0, then state is scored according to the heuristic. 
        if depth == 0:
            # Note that agent will always be different from cur_player in this fn call
            return state.heuristic(agent=self.agent)
         
        # If depth > 0, then state is evaluated as min of child states. 
        # Child states will be in a max layer.
         
        min_v = float('inf')
        for a in state.get_actions():
            # Evaluate child, which is in a Max layer. 
            child_state = state.take_action(a)
            v = self.max_valuation(child_state, alpha, beta, depth - 1)
            min_v = min(v, min_v)
            
            if min_v < alpha:
                return min_v
            
            beta = min(beta, min_v)
            
        return min_v

    def max_valuation(self, state, alpha, beta, depth):
        # The current player for states in a max layer is the agent. 
        # States are evaluated from the persective of the agent. 
        
        # Check to see if the current state is terminal. 
        if state.winner is None:
            pass
        elif state.winner == 0:
            return 0                        # Tied state
        elif state.winner == self.agent:
            return float('inf')             # Agent wins in this state
        else:
            return float('-inf')            # Agent loses in this state
        # The 2nd to last condition above should never be reached. 
        # Since we are in a max layer, oppo would be the last player to
        # have taken an action. That can't result in agent win.  
        
        # If the state is not terminal, we continue with valuation. 
        
        # If depth == 0, then state is scored according to the heuristic. 
        if depth == 0:
            # Note that agent will always be cur_player in this fn call
            return state.heuristic(agent=self.agent)
         
        # If depth > 0, then state is evaluated as max of child states. 
        # Child states will be in a min layer.
         
        max_v = float('-inf')
        for a in state.get_actions():
            # Evaluate child, which is in a Min layer. 
            child_state = state.take_action(a)
            v = self.min_valuation(child_state, alpha, beta, depth - 1)
            max_v = max(v, max_v)
            
            if max_v > beta:
                return max_v
            
            alpha = max(alpha, max_v)
            
        return max_v


class PolicyPlayer:
    
    def __init__(self, name, policy):
        self.name = name
        self.policy = policy.copy()
        
    def select_action(self, node):
        actions = node.get_actions()
        s = node.get_state()
        a = self.policy.get(s, None)
        if a not in actions:
            i = np.random.choice(len(actions))
            a = actions[i]
        return a


from IPython.display import clear_output
class HumanPlayer:
    
    def __init__(self, name, clear=True, show_actions=False):
        self.name = name
        self.clear = clear
        self.show_actions = show_actions
        
    def select_action(self, node):
        
        valid = False
        while valid == False:
            actions = node.get_actions()
            
            if self.show_actions:
                print('Valid Actions:', actions)
                print('Enter "quit" to end the game.')
                print()
                
            a = input('Please select an action:')
            print()
            
            try:  
                if a == 'quit':
                    return a   
                elif type(actions[0]) == int:
                    a = int(a)
                elif type(actions[0]) == tuple:
                    a = a.replace('(', '').replace(')', '')
                    a = tuple(int(x) for x in a.split(","))
            except:
                print('Action entered was not in correct format.')    
                continue
            
            if a not in actions:
                print('Selected action is not valid.')
                node.display()
            else:
                valid = True
        
        if self.clear:
            #time.sleep(0.5)
            clear_output()
            time.sleep(0.25)
            
        return a
    
        
