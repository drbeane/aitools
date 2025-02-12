from math import e
import numpy as np
import pandas as pd


count = 0

class Tree:

    def __init__(self, cond_level=100, seed=None, verbose=False):
        
        if seed is not None:
            np.random.seed(seed)

        while True:
            #print('FAILED')
            self.build_tree()
            self.assign_costs()
            self.find_solns()
            
            soln_list = list(self.solns.values())
            conditions = [
                self.all_unique, 
                len(np.unique(soln_list)) == 3, 
                self.solns['gbf'] != self.solns['ast'],  
                len(self.exp_order['ast']) > 4,
                len(self.exp_order['dfs']) <= 12,
                len(self.exp_order['ucs']) <= 12,
                len(self.exp_order['ast']) <= 12,
            ][:cond_level]
            
            if sum(conditions) == len(conditions):
                break
        
        return
        
        
    
    def build_tree(self):
        letters = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
        nodes = {x:{'cost':None, 'heur':None, 'children':[], 'parent':None} for x in letters}
        unused_letters = letters.copy()
        unused_letters.remove('A')
        level = [['A'], [], [], []]
            
        # 3 random nodes in level 1
        level[1] = sorted(list(np.random.choice(unused_letters, size=3, replace=False)))
        for x in level[1]: unused_letters.remove(x)
        # 7 random nodes in level 2
        level[2] = sorted(list(np.random.choice(unused_letters, size=7, replace=False)))
        # 15 random nodes in level 1
        for x in level[2]: unused_letters.remove(x)
        level[3] = sorted(list(unused_letters))
        
        #------------------------------
        # Set Children
        #------------------------------
        # Set children of root
        nodes['A']['children'] = level[1]
        
        # Assign nodes in lvl 2 to lvl 1 parents
        k = np.random.choice([0,1,2])  # Select one node to have 3 children
        temp = level[2].copy()
        for i, node in enumerate(level[1]):
            t = 3 if i == k else 2
            nodes[node]['children'] = temp[:t]
            temp = temp[t:]
        
        # Assign nodes in lvl 3 to lvl 2 parents
        j = np.random.choice([1,2,3,4])  # Select one L2 node to have no children
        options = [x for x in [0,1,2,3,4,5,6] if x != j]
        k = np.random.choice(options, size=3, replace=False) # Select 3 L2 nodes to have 3 children
        temp = level[3].copy()
        for i, node in enumerate(level[2]):
            if i == j:
                continue
            t = 3 if i in k else 2
            nodes[node]['children'] = temp[:t]
            temp = temp[t:]
        
        
        #------------------------------
        # Set Parents
        #------------------------------
        for N in letters:
            for C in nodes[N]['children']:
                nodes[C]['parent'] = N
        
        #------------------------------------------
        # Exploration Order for BFS and DFS
        #------------------------------------------
        bfs_order_full = sum(level, start=[])
        dfs_order_full = ['A']
        for L1_node in level[1][::-1]:
            dfs_order_full.append(L1_node)
            for L2_node in nodes[L1_node]['children'][::-1]:
                dfs_order_full.append(L2_node)
                for L3_node in nodes[L2_node]['children'][::-1]:
                    dfs_order_full.append(L3_node)
        
        
        #------------------------------
        # Set Goals
        #------------------------------
        while True:
            G1 = level[2][j]
            G2, G3 = np.random.choice(level[3][1:-2], size=2, replace=False)
            
            goals = sorted([G1, G2, G3])
            
            bfs_soln_idx = min([bfs_order_full.index(g) for g in goals])
            dfs_soln_idx = min([dfs_order_full.index(g) for g in goals])
            bfs_soln = bfs_order_full[bfs_soln_idx]
            dfs_soln = dfs_order_full[dfs_soln_idx]

            conditions = [
                bfs_soln != dfs_soln,
            ]
            
            if sum(conditions) == len(conditions):
                break
        
        bfs_exp_order = bfs_order_full[:bfs_soln_idx+1]
        dfs_exp_order = dfs_order_full[:dfs_soln_idx+1]
            
        self.solns = {'dfs':dfs_soln, 'bfs':bfs_soln}
        self.exp_order = {'dfs':dfs_exp_order, 'bfs':bfs_exp_order}
        
        self.level = level
        self.nodes = nodes
        self.goals = goals
        self.l2_goal = G1
    
    
    def assign_costs(self):
        import pandas as pdf
        nodes = self.nodes
        level = self.level
        
        cost_df = pd.DataFrame({
            'cost':np.zeros(26),
            'heur':np.zeros(26),
            'total':np.zeros(26),
        }, index=list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')).astype(int)
        
        
        #---------------------------------
        # Assign Actual Costs
        #---------------------------------
        while True:
            cost_options = list(range(5,85))
            
            # Level 0
            #Ac = np.random.choice([c for c in cost_options if c <= 15])
            cost_df.loc['A', 'cost'] = 0
            #cost_options.remove(Ac)
            
            # Level 1
            for N in level[1]:
                option_array = np.array(cost_options)
                options = option_array[option_array < 35]
                c = np.random.choice(options)
                cost_df.loc[N, 'cost'] = c
                cost_options.remove(c)

            # Level 2
            for N in level[2]:
                parent = nodes[N]['parent']
                par_cost = cost_df.loc[parent, 'cost']
                option_array = np.array(cost_options)
                options = option_array[(option_array >= par_cost + 8) & (option_array < 60)]
                c = np.random.choice(options)
                cost_df.loc[N, 'cost'] = c
                cost_options.remove(c)
            
            # Level 3
            for N in level[3]:
                parent = nodes[N]['parent']
                par_cost = cost_df.loc[parent, 'cost']
                option_array = np.array(cost_options)
                options = option_array[(option_array >= par_cost + 8)]
                #print(options)
                c = np.random.choice(options)
                cost_df.loc[N, 'cost'] = c
                cost_options.remove(c)
        
            goal_costs = [cost_df.loc[g, 'cost'] for g in self.goals]
            bfs_cost = cost_df.loc[self.solns['bfs'], 'cost']
            
            b1 = bfs_cost != min(goal_costs)
            c1, c2, c3 = goal_costs
            b2 = min([abs(c1 - c2), abs(c1 - c3), abs(c2 - c3)]) >= 3
            if b1 and b2:
                break
        
        
        def find_min_soln_cost(node):
            
            if node in self.goals:
                return cost_df.loc[node, 'cost']
            
            min_cost = 99
            children = nodes[node]['children']
            for c in children:
                if c in self.goals:
                    min_cost = min([min_cost, cost_df.loc[c, 'cost']])
                else:
                    temp = find_min_soln_cost(c)
                    min_cost = min([min_cost, temp])
            return min_cost
        
        #---------------------------------
        # Assign Heuristic Values
        #---------------------------------
        goal_costs = []
        for g in self.goals:
            cost_df.loc[g, 'total'] = cost_df.loc[g, 'cost']
            goal_costs.append(cost_df.loc[g, 'cost'])
            
        cost_df['max_heur'] = np.zeros(26).astype(int)
        
        letters = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
        
        for N in letters:
            cost = cost_df.loc[N, 'cost']
            cost_df.loc[N, 'max_heur'] = find_min_soln_cost(N) - cost
        
        cost_df = cost_df.sort_values('max_heur')
            
        found = False
        while not found:
            try:
                used_heur = []
                used_totals = goal_costs.copy()
                for N in cost_df.index.values:
                    if N in self.goals:
                        continue
                    
                    cost = cost_df.loc[N, 'cost']
                    options = range(5, cost_df.loc[N, 'max_heur'])
                    options = [x for x in options if x not in used_heur]
                    options = [x for x in options if x+cost not in used_totals]
                    options = [x for x in options if x+cost < 100]
                    
                    h = np.random.choice(options)
                    t = cost + h
                    
                    cost_df.loc[N, 'heur'] = h
                    cost_df.loc[N, 'total'] = t
                    used_heur.append(h)
                    used_totals.append(t)
                    found = True
            except:
                print('FAILED!')
                pass
                

        cost_df = cost_df.sort_index()
            
            
        self.cost_df = cost_df
        
        
    def find_solns(self):
        nodes = self.nodes
        
        sorted_cost_df = self.cost_df.sort_values('cost')
        
        # Find UCS Solution
        ucs_order_full = list(sorted_cost_df.index)
        ucs_soln_idx = min([ucs_order_full.index(g) for g in self.goals])
        ucs_soln = ucs_order_full[ucs_soln_idx]
                       

        ucs_exp_order = ucs_order_full[:ucs_soln_idx+1]
        
        
        # Find GBF Solution
        frontier = [('A', self.cost_df.loc['A', 'heur'])]
        gbf_exp_order = []
        while True:
            sel = frontier.pop(0)
            node = sel[0]
            gbf_exp_order.append(node)
            if node in self.goals: break
                
            for c in nodes[node]['children']:
                frontier.append((c, self.cost_df.loc[c, 'heur']))
            frontier = sorted(frontier, key=lambda x : x[1])
        gbf_soln = gbf_exp_order[-1]
        
        # Find AST Solution
        priority = self.cost_df.loc['A', 'total']
        frontier = [('A', priority)]
        ast_exp_order = []
        while True:
            sel = frontier.pop(0)
            node = sel[0]
            ast_exp_order.append(node)
            if node in self.goals: break
                
            for c in nodes[node]['children']:
                priority = self.cost_df.loc[c, 'total']
                frontier.append((c, priority))
            frontier = sorted(frontier, key=lambda x : x[1])
        ast_soln = ast_exp_order[-1]
        
        
        dfs_soln = self.solns['dfs']
        bfs_soln = self.solns['bfs']
        soln_list = [dfs_soln, bfs_soln, ucs_soln, gbf_soln, ast_soln]
        
        self.solns['ucs'] = ucs_soln
        self.solns['gbf'] = gbf_soln
        self.solns['ast'] = ast_soln
        
        
        
        dfs_exp_order = self.exp_order['dfs'] 
        bfs_exp_order = self.exp_order['bfs'] 
        
        self.exp_order['ucs'] = ucs_exp_order
        self.exp_order['gbf'] = gbf_exp_order
        self.exp_order['ast'] = ast_exp_order
        
        letters = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
        
        # Build Distrctors
        dfs_distractor = dfs_exp_order.copy()
        bfs_distractor = bfs_exp_order.copy()
        ucs_distractor = ucs_exp_order.copy()
        gbf_distractor = gbf_exp_order.copy()
        ast_distractor = ast_exp_order.copy()
        d_list = [dfs_distractor, bfs_distractor, ucs_distractor, gbf_distractor, ast_distractor]
        a_list = [dfs_exp_order, bfs_exp_order, ucs_exp_order, gbf_exp_order, ast_exp_order]
        for dist, ans in zip(d_list, a_list):
            while True:
                n_replace = len(ans) - 3 if len(ans) > 3 else 1
                temp = [x for x in letters if x not in ans[:2] + ans[-1:]]
                new_mid = np.random.choice(temp, size=n_replace)
                start = 2 if len(ans) > 3 else 1
                for i, v in enumerate(new_mid): dist[start+i] = v
                if dist != ans: break
        
        self.a_list = a_list                
        self.d_list = d_list
        self.distractors = d_list
                        
        # Loop to check uniqueness of answers and distractors
        all_unique = True
        options = d_list + a_list
        for i in range(9):
            for j in range(i+1, 10):
                if options[i] == options[j]: 
                    all_unique = False
                       
        self.all_unique = all_unique      
        
           
   
    def build_tree_SAVE(self):
        letters = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
        nodes = {x:{'cost':0, 'heur':99, 'children':[]} for x in letters}
        unused_letters = letters.copy()
        unused_letters.remove('A')
        level = [['A'], [], [], []]
            
        level[1] = sorted(list(np.random.choice(unused_letters, size=3, replace=False)))
        for x in level[1]: unused_letters.remove(x)
        level[2] = sorted(list(np.random.choice(unused_letters, size=7, replace=False)))
        for x in level[2]: unused_letters.remove(x)
        level[3] = sorted(list(unused_letters))
        
        # Set Children
        nodes['A']['children'] = level[1]
        
        k = np.random.choice([0,1,2])
        temp = level[2].copy()
        for i, node in enumerate(level[1]):
            t = 3 if i == k else 2
            nodes[node]['children'] = temp[:t]
            temp = temp[t:]
            
        k = np.random.choice([0,1,2,3,4,5,6])
        temp = level[3].copy()
        for i, node in enumerate(level[2]):
            t = 3 if i == k else 2
            nodes[node]['children'] = temp[:t]
            temp = temp[t:]
        
        
        #------------------------------------------
        # Exploration Order for BFS and DFS
        #------------------------------------------
        bfs_order_full = sum(level, start=[])
        dfs_order_full = ['A']
        for L1_node in level[1][::-1]:
            dfs_order_full.append(L1_node)
            for L2_node in nodes[L1_node]['children'][::-1]:
                dfs_order_full.append(L2_node)
                for L3_node in nodes[L2_node]['children'][::-1]:
                    dfs_order_full.append(L3_node)
        
        
        #------------------------------------------
        # Set Goals
        # Also find solutions for BFS and DFS
        #------------------------------------------
        while True:
            G1 = np.random.choice(level[2][2:-2])
            G2 = np.random.choice(level[3][:-3])
            G3 = np.random.choice(level[2][2:-2] + level[3][:-3])
            goals = sorted([G1, G2, G3])
            
            bfs_soln_idx = min([bfs_order_full.index(g) for g in goals])
            dfs_soln_idx = min([dfs_order_full.index(g) for g in goals])
            bfs_soln = bfs_order_full[bfs_soln_idx]
            dfs_soln = dfs_order_full[dfs_soln_idx]

            
            conditions = [
                bfs_soln != dfs_soln,
                G1 != G2, G1 != G3, G2 != G3,
                G2 not in nodes[G1]['children'], 
                G3 not in nodes[G1]['children'],
                G2 not in nodes[G3]['children'],
            ]
            
            if sum(conditions) == len(conditions):
                break
        
        self.level = level
        self.nodes = nodes
        

    def get_display_str(self):
        level = self.level
        nodes = self.nodes

        #---------------------------
        # Build Bottom Half
        #---------------------------
        lvl2_string = ''
        conn_2_3 = ''
        lvl3_string = ''
        for node in level[2]:
            children = nodes[node]['children']
            
            if len(children) == 0:
                lvl2_string += f'{node}   '
                conn_2_3 += ' '*4
                lvl3_string += ' '*4
            elif len(children) == 2:
                lvl2_string += f' {node}    '
                lvl3_string += ' '.join(children) + '   '
                conn_2_3 += '┌┴┐   '
            else:
                lvl2_string += f'  {node}     '
                lvl3_string += ' '.join(children) + '   '
                conn_2_3 += '┌─┼─┐   '
            
        #---------------------------
        # Build Top Half
        #---------------------------
        blank = list(' '*len(lvl3_string))
        lvl0_array = np.array(blank)
        lvl1_array = np.array(blank)
        conn_0_1_array = np.array(blank)
        conn_1_2_array = np.array(blank)
        
        for i, node in enumerate(level[1]):
            children = nodes[node]['children']
            m0 = lvl2_string.index(children[0])
            m1 = lvl2_string.index(children[1])
            m2 = lvl2_string.index(children[-1])
            
            if len(children) == 2:
                loc = int((m0 + m2)/2)
                conn_char = '┴'
            elif len(children) == 3:
                loc = m1
                conn_char = '┼'
                
            lvl1_array[loc] = node           # Place L1 Node
            if i == 1: lvl0_array[loc] = 'A'  # Place L0 Node
            
            # Level 1-2 Connectors   
            conn_1_2_array[m0:m2] = '─'
            conn_1_2_array[m0] = '┌'
            conn_1_2_array[m2] = '┐'
            conn_1_2_array[loc] = conn_char
            
            # Level 0-1 Connectors
            if i == 0: 
                conn_0_1_array[loc:] = '─'
                conn_0_1_array[loc] = '┌'
            if i == 1:
                conn_0_1_array[loc] = '┼'
            if i == 2:
                conn_0_1_array[loc:] = ' '
                conn_0_1_array[loc] = '┐'
            
        lvl0_string = ''.join(lvl0_array)
        lvl1_string = ''.join(lvl1_array)
        conn_0_1 = ''.join(conn_0_1_array)
        conn_1_2 = ''.join(conn_1_2_array)
        
        display_str = '\n'.join([
            lvl0_string, conn_0_1, lvl1_string, conn_1_2, lvl2_string, conn_2_3, lvl3_string
        ])
        display_str = display_str.strip('\n')
        
        display_str_html = '<span style="font-size: 20px; font-family: monospace, monospace; line-height: 20px;">\n'
        display_str_html += display_str.replace(' ', '&nbsp;').replace('\n', '<br />')
        display_str_html += '</span>'
        
        #print(L0, H1, L1, H2, L2, H3, L3, sep='\n')
        return display_str, display_str_html
            
        
        

if __name__ == '__main__':#
    print('-'*32)
    sd = np.random.choice(range(1000))
    #sd=115
    print(sd)
    tree = Tree(cond_level=100, seed=sd, verbose=True)
    tree_str, tree_html = tree.get_display_str()
    print(tree_str)
    print()
    print('Goals:', tree.goals)
    print()
    goal_costs = [tree.cost_df.loc[g, 'cost'] for g in tree.goals]
    print(goal_costs)
    #print(tree.cost_df)
    #print()
    
    
    
    
    c1 = np.sum(tree.cost_df.cost < 20)
    c2 = np.sum((tree.cost_df.cost > 20) &(tree.cost_df.cost < 30))
    c3 = np.sum((tree.cost_df.cost > 30) &(tree.cost_df.cost < 40))
    c4 = np.sum((tree.cost_df.cost > 40) &(tree.cost_df.cost < 50))
    c5 = np.sum((tree.cost_df.cost > 50) &(tree.cost_df.cost < 60))
    c6 = np.sum((tree.cost_df.cost > 60) &(tree.cost_df.cost < 70))
    c7 = np.sum((tree.cost_df.cost > 70) &(tree.cost_df.cost < 80))
    c8 = np.sum((tree.cost_df.cost > 80) &(tree.cost_df.cost < 90))
    c9 = np.sum((tree.cost_df.cost > 90) &(tree.cost_df.cost < 100))
    
    print(c1, c2, c3, c4, c5, c6, c7, c8, c9)
    
    
    
    
    
    #for k, v in tree.exp_order.items():
    #    print(k)
    #    print(v)
    #    print()
    
    
'''
                   A                          
    ┌──────────────┼───────────────┐        
    T              C               D        
 ┌──┴──┐     ┌─────┼─────┐      ┌──┴───┐    
 E     H     W     X     F      I      L    
┌┴┐   ┌┴┐   ┌┴┐   ┌┴┐   ┌┴┐   ┌─┼─┐   ┌┴┐   
G E   O H   U W   M X   N F   J Q I   V L   
'''