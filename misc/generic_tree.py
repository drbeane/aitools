from math import e
import numpy as np
import pandas as pd


count = 0

class Tree:

    def __init__(self, cond_level=100, seed=None, verbose=False):
        
        if seed is not None:
            np.random.seed(seed)

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
        
        
        #-----------------------------------------
        # Set Path Costs and Heuristic Values
        # And find UCS, GBF, and AST solns
        #-----------------------------------------
        while True:
            #for n in letters: nodes[n]['cost'] = 0
            
            # Set Path Costs
            for lvl in level[:3]:
                for node in lvl:
                    base = nodes[node]['cost']
                    for child in nodes[node]['children']:
                        d = np.random.choice(range(5,50))
                        nodes[child]['cost'] = base + d

            # Determine Solution Costs
            soln_costs = [nodes[g]['cost'] for g in goals]

            # Set Heuristic Values
            huer_values_used = []
            for g in goals: nodes[g]['heur'] = 0
            nodes['A']['heur'] = min(soln_costs) - 5
            for node in level[1]:
                max_cost = 99
                child_soln_found = False
                for c2 in nodes[node]['children']: # Explore L2 children
                    if child_soln_found: break
                    if c2 in goals:
                        max_cost = min(max_cost, nodes[c2]['cost'])
                        child_soln_found = True
                        break
                    
                    for c3 in nodes[c2]['children']: # Explore L3 children
                        if c3 in goals:
                            max_cost = min(max_cost, nodes[c3]['cost'])
                            child_soln_found = True
                            break
                options = list(range(1, max_cost - nodes[node]['cost']))
                for h in huer_values_used:
                    if h in options: options.remove(h)
                h_val = np.random.choice(options)
                huer_values_used.append(h_val)
                nodes[node]['heur'] = h_val
            
            for node in level[2]:
                if node in goals: continue
                max_cost = 99
                for c3 in nodes[node]['children']: # Explore L3 children
                    if c3 in goals:
                        max_cost = min(max_cost, nodes[c3]['cost'])
                        break
                
                options = list(range(1, max_cost - nodes[node]['cost']))
                for h in huer_values_used:
                    if h in options: options.remove(h)
                try:
                    h_val = np.random.choice(options)
                except:
                    h_val = 1
                huer_values_used.append(h_val)
                nodes[node]['heur'] = h_val
            
            # Create Data Structres for Costs and Heuristics
            costs = [nodes[x]['cost'] for x in letters]           
            heur = [nodes[x]['heur'] for x in letters]
            cost_df = pd.DataFrame(dict(cost=costs, heur=heur), index=letters)            
            
            # Change 99 heuristics
            non_trivial_hvals = [x for x in cost_df.heur if x not in [0,99]]
            idx_99 = np.where(cost_df.heur.values == 99)[0]
            rand_heur = np.random.choice(
            range(max(non_trivial_hvals) + 1, 100), size=len(idx_99)
                )
            for i,h in zip(idx_99, rand_heur):
                cost_df.heur.values[i] = h
            
            cost_df['total'] = cost_df.cost + cost_df.heur
            sorted_cost_df = cost_df.sort_values('cost')
            
            # Find UCS Solution
            ucs_order_full = list(sorted_cost_df.index)
            ucs_soln_idx = min([ucs_order_full.index(g) for g in goals])
            ucs_soln = ucs_order_full[ucs_soln_idx]
            
            # Node expansion orders of uniformed algs
            bfs_exp_order = bfs_order_full[:bfs_soln_idx+1]
            dfs_exp_order = dfs_order_full[:dfs_soln_idx+1]
            ucs_exp_order = ucs_order_full[:ucs_soln_idx+1]
            
            # Find GBF Solution
            frontier = [('A', nodes['A']['heur'])]
            gbf_exp_order = []
            while True:
                sel = frontier.pop(0)
                node = sel[0]
                gbf_exp_order.append(node)
                if node in goals: break
                    
                for c in nodes[node]['children']:
                    frontier.append((c, nodes[c]['heur']))
                frontier = sorted(frontier, key=lambda x : x[1])
            gbf_soln = gbf_exp_order[-1]
            
            # Find AST Solution
            priority = nodes['A']['heur'] + nodes['A']['cost']
            frontier = [('A', priority)]
            ast_exp_order = []
            while True:
                sel = frontier.pop(0)
                node = sel[0]
                ast_exp_order.append(node)
                if node in goals: break
                    
                for c in nodes[node]['children']:
                    priority = nodes[c]['heur'] + nodes[c]['cost']
                    frontier.append((c, priority))
                frontier = sorted(frontier, key=lambda x : x[1])
            ast_soln = ast_exp_order[-1]
            soln_list = [dfs_soln, bfs_soln, ucs_soln, gbf_soln, ast_soln]
            
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
                            
                            
            # Loop to check uniques of answers and distractors
            all_unique = True
            options = d_list + a_list
            for i in range(9):
                for j in range(i+1, 10):
                    if options[i] == options[j]: 
                        all_unique = False
            
            # Set Conditions    
            conditions = [
                all_unique, # 1
                len(cost_df.cost.unique()) == 26, #2 
                len(cost_df.total.unique()) == 26, # 3
                len(np.unique(non_trivial_hvals)) == len(non_trivial_hvals), #4 
                max(soln_costs) < 2 * min(soln_costs), # 5
                len(np.unique(soln_list)) == 3, # 6
                max(costs) < 100,       # 7
                gbf_soln != ast_soln,   # 8 
            ][:cond_level]
            
            if sum(conditions) == len(conditions):
                break
        
        self.solns = dict(
            dfs=dfs_soln, bfs=bfs_soln, ucs=ucs_soln, gbf=gbf_soln, ast=ast_soln
        )
        self.exp_order = dict(
            dfs=dfs_exp_order, bfs=bfs_exp_order, ucs=ucs_exp_order,
            gbf=gbf_exp_order, ast=ast_exp_order
        )
        self.cost_df = cost_df
        self.level = level
        self.nodes = nodes
        self.goals = goals
        self.distractors = d_list
        
        

    def get_display_str(self):
        
        #     L0
        # H1, L1
        # H2, L2
        # H3, L3

        L2 = ''
        H3 = ''
        L3 = ''

        for n2 in self.level[2]:
            children = self.nodes[n2]['children']
            for n3 in children:
                L3 += f'{n3} '
            L3 += '  ' 

            if len(children) == 2: 
                H3 += '┌┴┐   '
                L2 += f' {n2}    '
            else: 
                H3 += '┌─┼─┐   '
                L2 += f'  {n2}     '
        L3 = L3.strip()
        
        L2_idx = np.where(np.array(list(L2)) != ' ')[0]
                
        L0 = np.array([' '] * len(L3))
        H1 = np.array([' '] * len(L3))
        L1 = np.array([' '] * len(L3))
        H2 = np.array([' '] * len(L3))
        
        for i, node in enumerate(self.level[1]):
            n = len(self.nodes[node]['children'])
            child_locs = L2_idx[:n]
            L2_idx = L2_idx[n:]
            m0, m1 = min(child_locs), max(child_locs)
            
            # Build H2
            H2[m0:m1] = '─'; H2[m0] = '┌';  H2[m1] = '┐'
            j = int((m0 + m1)/2) if n==2 else child_locs[1]
            conn = '┴' if n==2 else '┼'
            H2[j] = conn
            
            L1[j] = node
            
            if i == 0: H1[j] = '┌'; H1[(j+1):] = '─'
            if i == 1: H1[j] = '┼'; L0[j] = 'A'
            if i == 2: H1[j] = '┐'; H1[(j+1):] = ' '
            
        

        L0, H1, L1, H2 = ''.join(L0), ''.join(H1), ''.join(L1), ''.join(H2)
        
        display_str = '\n'.join([L0, H1, L1, H2, L2, H3, L3])
        display_str = display_str.strip('\n')
        
        display_str_html = '<span style="font-size: 20px; font-family: monospace, monospace; line-height: 20px;">\n'
        display_str_html += display_str.replace(' ', '&nbsp;').replace('\n', '<br />')
        display_str_html += '</span>'
        
        #print(L0, H1, L1, H2, L2, H3, L3, sep='\n')
        return display_str, display_str_html

        l1_idx = []           
        for i, n1 in enumerate(level_1):

            l2_children = nodes[n1]['children']
            n = len(l2_children)
            child_indices = bar_indices[:n]
            print(child_indices)
            m0 = min(child_indices)
            m1 = max(child_indices)

            hor_2_list[m0] = '┌'
            hor_2_list[m1] = '┐'
            for j in range(m0+1, m1):
                hor_2_list[j] = '─'
            if n == 3:
                j = child_indices[1]
                hor_2_list[j] = '┼'
                if i==1: l0_string_list[j] = 'A'
                l1_string_list[j] = n1
                hor_1_list[j] = ['┌', '┼', '┐'][i]
                

            else:
                j = int((m0 + m1)/2)
                hor_2_list[j] = '┴'
                if i==1: l0_string_list[j] = 'A'
                l1_string_list[j] = n1
                hor_1_list[j] = ['┌', '┼', '┐'][i]

            bar_indices = bar_indices[n:]


        a = hor_1_list.index('┌')
        b = hor_1_list.index('┐')
        for i in range(a, b):
            if hor_1_list[i] == ' ':
                hor_1_list[i] = '─'

        hor_1 = ''.join(hor_1_list)
        hor_2 = ''.join(hor_2_list)
        l0_string = ''.join(l0_string_list)
        l1_string = ''.join(l1_string_list)

        print(l0_string)
        print(hor_1)
        print(l1_string)
        print(hor_2)
        print(l2_string)
        print(hor_3)
        print(l3_string)

if __name__ == '__main__':#
    print('-'*32)
    tree = Tree(cond_level=1, seed=452, verbose=True)
    tree_str, tree_html = tree.get_display_str()
    print(tree_str)
    print()
    print(tree.goals)
    print()
    print(tree.cost_df)
    
    
'''
                   A                          
    ┌──────────────┼───────────────┐        
    T              C               D        
 ┌──┴──┐     ┌─────┼─────┐      ┌──┴───┐    
 E     H     W     X     F      I      L    
┌┴┐   ┌┴┐   ┌┴┐   ┌┴┐   ┌┴┐   ┌─┼─┐   ┌┴┐   
G E   O H   U W   M X   N F   J Q I   V L   
'''