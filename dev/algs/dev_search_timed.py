def general_search(root, alg, time_limit=120, display_results=True, 
                   update_rate=None, timer=None, track_mem=False, **kwargs):
    '''
    Performs a search. 
    
    Arguments: 
        root -- Root node instance of the enviroment being searched. 
        alg -- String indicating search algorithm. Options are: 
            'DFS', 'BFS', 'UCS', 'GBF', and 'AST'
        time_limit -- Maximum time allowed for the search. 
        update_rate -- Frequency of updates while search is being conducted.
            An update is displayed every update_rate nodes that are checked. 
            If None, no updated displayed. 
        kwargs -- allows for additional arguments to be passed down to the heuristic. 
    '''    
    import time
    import heapq
    
    t0 = time.time()                   # Start the search timer
    
    frontier = [(0,root.copy())]       # Create the frontier
    nodes_visited = set()              # Set to store the visited nodes

    log = {
        'nodes_seen':1,       # Counter for number of nodes seen, including skipped nodes
        'nodes_checked':0,    # Counter for number of nodes checked
        'nodes_skipped':0,    # Counter for number of nodes skipped
        'nodes_queued':1,     # Counter for number of nodes queued (some might later be skipped)                   
        'time':0,             # Stores total time taken for search (in seconds)
        'frontier_size':[],   # List to track size of frontier
        'frontier_mem':[]     # List to track size of frontier
    }

    #timer = Timer()

    # Continue search for as long as there are nodes in the frontier
    while len(frontier) > 0:
        
        timer.start('popping node')
        priority, cur_node = heapq.heappop(frontier)  # Get a new node from the frontier
        timer.stop('popping node')
        
        # Skip the current node if its state has been previously seen.
        timer.start('getting state_id')
        cur_node_id = cur_node.get_state_id()
        timer.stop('getting state_id')
        
        timer.start('checking visited')
        cv = cur_node_id in nodes_visited
        timer.stop('checking visited')
        if cv:
            log['nodes_skipped'] += 1
            continue
        
        
        # If the current node has not been previously visited, add it to visited. 
        # Some environments have no possibility of returning to previously visited states.
        # Nodes for those environemnts will have 'None' as their ID. 
        timer.start('adding visited')
        if cur_node_id is not None:
            nodes_visited.add(cur_node_id)
        timer.stop('adding visited')
        
        # Increment the nodes_checked counter.
        log['nodes_checked'] += 1
        
        # Add the new frontier size to the log. 
        fs = len(frontier)
        log['frontier_size'].append(fs)
        
        # Determine Memory Requirements of Frontier. 
        #if track_mem:
        #    mem = 0
        #    for node in frontier:
        #        mem += node[1].get_mem_req()
        #    log['frontier_mem'].append(mem)
            
        
        # Check to see if current state is a goal state
        # If it is, then return print results and the current node
        timer.start('check solved')
        if cur_node.check_solved():
            log['time'] = time.time() - t0
            
            # Display the results of the search. 
            if display_results:
                print(f'[{alg}] Solution found.')
                print(f'{log["nodes_seen"]} nodes seen.') 
                print(f'{log["nodes_skipped"]} nodes skipped')
                print(f'{log["nodes_checked"]} nodes expanded.')
                print(f'{len(frontier)} nodes remaining in frontier.')
                print(f'{round(log["time"], 2)} seconds elapsed.')
                print(cur_node.soln_info())
                print()
            
            return cur_node, log
        timer.stop('check solved')
        
        # If current node is not a solution, then expand it by looping over all available actions. 
        timer.start('get actions')
        actions = cur_node.get_actions()    
        timer.stop('get actions')
        
        timer.start('looping over actions')
        for a in actions:
            log['nodes_seen'] += 1
            timer.start('ACTION - applying action')
            child = cur_node.take_action(a)     # Get child node
            timer.stop('ACTION - applying action')
            timer.start('ACTION - get child id')
            child_id = child.get_state_id()     # Get the id of the child
            timer.stop('ACTION - get child id')
            
            # If we have previously visited the child state, we can skip it. 
            timer.start('ACTION - check if child visited')
            cv = child_id in nodes_visited
            timer.stop('ACTION - check if child visited')
            if cv:
                log['nodes_skipped'] += 1
            else:
                # If the child has not been visited before, then we need to add it to the frontier
                log['nodes_queued'] += 1
                                
                timer.start('ACTION - calculate priority')
                # Calculate the priority to be assigned to the child 
                if alg == 'DFS':
                    priority = -log['nodes_queued']
                elif alg == 'BFS':
                    priority = log['nodes_queued']
                elif alg == 'UCS':
                    priority = child.path_cost()
                elif alg == 'GBF':
                    priority = child.heuristic(alg='GBF')
                elif alg == 'AST':
                    priority = child.path_cost() + child.heuristic(alg='AST')
                timer.stop('ACTION - calculate priority')        
                
                # Add the child to the frontier
                timer.start('ACTION - add child to frontier')
                heapq.heappush(frontier, (priority, child))
                timer.stop('ACTION - add child to frontier')
        timer.stop('looping over actions')
        
        # Print a status message every update_rate iterations
        if update_rate is not None: 
            if log["nodes_checked"] % update_rate == 0:
                msg = f'{log["nodes_seen"]} seen. {log["nodes_checked"]} checked. '
                msg += f'{len(frontier)} in frontier. {log["nodes_skipped"]} skipped.'
                
                if track_mem:
                    mem = 0
                    for node in frontier:
                        mem += node[1].get_mem_req()
                        log['frontier_mem'].append(mem)
                    msg += f' Size of Frontier: {mem // 1024**2} MB' 
                
                print(msg)
        
        
        # Terminate search if max iterations has been reached 
        log['time'] = time.time() - t0
        if log['time'] >= time_limit:
            break
    
    # Print a message explaining that no solution was found. 
    log['time'] = time.time() - t0
    if display_results:
        print(f'[{alg}] No solution was found.')
        if log['time'] >= time_limit:
            print('Time limit reached.')
        print(f'{log["nodes_seen"]} nodes seen.') 
        print(f'{log["nodes_skipped"]} nodes skipped')
        print(f'{log["nodes_checked"]} nodes expanded.')
        print(f'{len(frontier)} nodes remaining in frontier.')
        print(f'{round(log["time"], 2)} seconds elapsed.')
        
        print()
    
    
    
    
    return None, log



##########################################################################
# The functions below are wrappers for the general_search() function
# They can be called to apply specific search algorithms. 
##########################################################################

def depth_first_search(env, time_limit=120, display_results=True, update_rate=None, **kwargs):
    return general_search(env, 'DFS', time_limit, display_results, update_rate, **kwargs)

def breadth_first_search(env, time_limit=120, display_results=True, update_rate=None, **kwargs):
    return general_search(env, 'BFS', time_limit, display_results, update_rate, **kwargs)
    
def uniform_cost_search(env, time_limit=120, display_results=True, update_rate=None, **kwargs):
    return general_search(env, 'UCS', time_limit, display_results, update_rate, **kwargs)

def greedy_best_first(env, time_limit=120, display_results=True, update_rate=None, **kwargs):
    return general_search(env, 'GBF', time_limit, display_results, update_rate, **kwargs)

def astar_search(env, time_limit=120, display_results=True, update_rate=None, **kwargs):
    return general_search(env, 'AST', time_limit, display_results, update_rate, **kwargs)