import numpy as np


class abstract_minimax():
    
    def __init__(self, mode='max', seed=None, check_abp=False, verbose=False):
        
        if seed is not None:
            np_state = np.random.get_state()
            np.random.seed(seed)
        
        while True:
            #print('?')
            
            while True:
                self.values = np.random.choice(range(1, 99), size=12, replace=False)
                sorted = self.values.copy()
                sorted.sort()
                diffs = sorted[1:] - sorted[:-1]
                min_diff = diffs.min()
                if min_diff >= 5:
                    break
            
            self.set_values('min')
            self.set_values('max')
            
            self.root_value = self.max_root_value if mode=='max' else self.min_root_value
            
            if check_abp == False:
                break
            
            self.check_pruning('min')
            self.check_pruning('max')
            
            self.select_values_for_mode(mode)
            
            overlap = False
            contained = True
            for v in self.pruned_step_23:
                if v in self.pruned_step_12:
                    overlap = True
                else:
                    contained = False
        
            n12 = len(self.pruned_step_12)
            n23 = len(self.pruned_step_23)        
            n_wrong = len(self.min_pruned) if mode == 'max' else len(self.max_pruned) 
            
            max_min_overlap = self.min_pruned.intersection(self.max_pruned)
            #---------------------------
            # Accepted Scenarios:
            #---------------------------
                       
            # Scenario 1 
            # 1 at L2  -- 1 or 2 at L3 -- no overlap -- min/max solns share at most 1
            b1 = (n12 == 2) and (n23 > 0) and (not overlap) and (len(max_min_overlap) <= 1) and (n_wrong >= 2)
            
            # Scenario 2
            # 2 at L2 -- 1 or 2 at L3 -- not contained -- min/max solns share at most 2
            b2 = (n12 == 4) and (n23 in [1,2]) and (not contained) and (len(max_min_overlap) <= 2) and (n_wrong >= 2)
             
            if b1 or b2:
                self.set_answers_and_distractors()
                break
        
        
                
        if verbose:
            self.print_values()
            print()
            if check_abp:    
                print(self.pruned_step_12)
                print(self.pruned_step_23)
                print(self.pruned)
                print()
                print(f'Max: {self.max_pruned}')
                print(f'Min: {self.min_pruned}')

            print(self.answer)
            
        if seed is not None:
            np_state = np.random.get_state()


    def set_answers_and_distractors(self):
        
        letters = list('ABCDEFGHIJKL')
        self.answer = []
        for i in range(12):
            if self.values[i] in self.pruned:
                self.answer.append(letters[i])
        self.answer = ' '.join(self.answer)
        
        self.answer_wrong = []
        for i in range(12):
            if self.values[i] in self.pruned_wrong:
                self.answer_wrong.append(letters[i])
        self.answer_wrong = ' '.join(self.answer_wrong)
        
        distractors = [
            'D K L', 'D G H', 'D G H K L', 'H K L',
            'G H L', 'D H K L', 'D G H L',
        ]
        distractors.remove(self.answer)
        if self.answer_wrong not in distractors:
            distractors.append(self.answer_wrong)
        
        if len(distractors) == 6:
            distractors.append('D H L')
                    
        self.distractors = distractors

    def select_values_for_mode(self, mode):
        if mode == 'max':
            self.l1_values = self.max_l1_values
            self.l2_values = self.max_l2_values
            self.root_value = self.max_root_value
            self.pruned_step_12 = self.max_pruned_12
            self.pruned_step_23 = self.max_pruned_23
            self.pruned = self.max_pruned
            self.pruned_wrong = self.min_pruned
        else:
            self.l1_values = self.min_l1_values
            self.l2_values = self.min_l2_values
            self.root_value = self.min_root_value
            self.pruned_step_12 = self.min_pruned_12
            self.pruned_step_23 = self.min_pruned_23
            self.pruned = self.min_pruned
            self.pruned_wrong = self.max_pruned
      
    def set_values(self, mode):
            root_value = 0
            l1_values = []
            l2_values = []
            
            # Eval Level 2
            for i in range(6):
                child_values = self.values[(i*2):(i*2+2)]
                v = np.max(child_values) if mode=='max' else np.min(child_values)
                l2_values.append(v)

            # Eval Level 1
            for i in range(3):
                child_values = l2_values[(i*2):(i*2+2)]
                v = np.min(child_values) if mode=='max' else np.max(child_values)
                l1_values.append(v)

            root_value = np.max(l1_values) if mode=='max' else np.min(l1_values)
            
            if mode == 'max':
                self.max_l1_values = l1_values
                self.max_l2_values = l2_values
                self.max_root_value = root_value
            else:
                self.min_l1_values = l1_values
                self.min_l2_values = l2_values
                self.min_root_value = root_value

    def check_pruning(self, mode):
        
        if mode == 'max':
            values = self.values
            l2_values = self.max_l2_values
            l1_values = self.max_l1_values
        else:
            values = self.values
            l2_values = self.min_l2_values
            l1_values = self.min_l1_values
        
        # Check for level 2-3 pruning
        nodes_pruned_step_23 = []
        for i in [1, 3, 5]:   # Loop over 2nd children in L2
            sib_value = l2_values[i-1]
            child_1_value = values[2*i]
            child_2_value = values[2*i+1]
            
            if mode == 'max' and sib_value  <= child_1_value:
                nodes_pruned_step_23.append(child_2_value)
            if mode == 'min' and sib_value  >= child_1_value:
                nodes_pruned_step_23.append(child_2_value)
        
        # Check for level 1-2 pruning:
        nodes_pruned_step_12 = []
        for i in [1,2]:
            sib_values = l1_values[:i]
            child_1_value = l2_values[2*i]
            child_2_value = l2_values[2*i+1]
            
            pruned = False
            if mode == 'max' and child_1_value <= np.max(sib_values):
                pruned = True
                #print(f'{child_2_value} pruned!')
            if mode == 'min' and child_1_value >= np.min(sib_values):
                pruned = True
                #print(f'{child_2_value} pruned!')
            if pruned:
                v1 = values[i*4 + 2]
                v2 = values[i*4 + 3]
                nodes_pruned_step_12.append(v1)
                nodes_pruned_step_12.append(v2)
        
        pruned = set(nodes_pruned_step_12).union(set(nodes_pruned_step_23))
        
        if mode == 'max':
            self.max_pruned_12 = nodes_pruned_step_12.copy()
            self.max_pruned_23 = nodes_pruned_step_23.copy()
            self.max_pruned = pruned
        if mode == 'min':
            self.min_pruned_12 = nodes_pruned_step_12.copy()
            self.min_pruned_23 = nodes_pruned_step_23.copy()
            self.min_pruned = pruned


    def print_values(self):
        
        L3 = ''
        for i, v in enumerate(self.values):
            if i % 2 == 0: L3 += f'{v:>3}   '
            else: L3 += f'{v:>3}  '
        
        L2 = '   '
        for i, v in enumerate(self.l2_values):
            if i % 2 ==  0: L2 += f'{v:>3}        '
            else: L2 += f'{v:>3}        '
        
        L1 = '        '
        for i, v in enumerate(self.l1_values):
            if i % 2 ==  0: L1 += f'{v:>3}                   '
            else: L1 += f'{v:>3}                   '
        
        L0 = f'                              {self.root_value:>3}' 
        
        C01 = '          ┌─────────────────────┼─────────────────────┐ '
        C12 = '     ' + '┌────┴─────┐          ' * 3
        C23 = '  ' + '┌──┴──┐    ' * 6
        names = ('  A     B    C     D    E     F    G     H    I     J    K     L')
        
        print(L0.rstrip())
        print(C01.rstrip())
        print(L1.rstrip())
        print(C12.rstrip())
        print(L2.rstrip())
        print(C23.rstrip())
        print(L3.rstrip())
        print(names)


if __name__ == '__main__':
    
    mm = abstract_minimax(mode='max', check_abp=True, verbose=True, seed=None)
    #found = []
    #for i in range(10000):
    #    mm = abstract_minimax(mode='max', check_abp=True, verbose=False, seed=None)
    #    if mm.answer not in found:
    #        found.append(mm.answer)
    #        print(mm.answer)
            
    #mm.print_values()