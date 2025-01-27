import numpy as np


class abstract_minimax():
    
    def __init__(self, mode='max'):
        self.values = np.random.choice(range(-50, 51), size=12, replace=False)
        
        self.values = [-2, 1, 4, 8, -8, -6, -9, -8, 6, -7, 9, -5]
        
        self.root_value = 0
        self.l1_values = []
        self.l2_values = []
        
        # Eval Level 2
        for i in range(6):
            child_values = self.values[(i*2):(i*2+2)]
            v = np.max(child_values) if mode=='max' else np.min(child_values)
            self.l2_values.append(v)

        # Eval Level 1
        for i in range(3):
            child_values = self.l2_values[(i*2):(i*2+2)]
            v = np.min(child_values) if mode=='max' else np.max(child_values)
            self.l1_values.append(v)

        self.root_value = np.max(self.l1_values) if mode=='max' else np.min(self.l1_values)
        self.print_values()
        print()
        
        
        def min_valuation(idx, level, a, b):
            if level == 3: 
                #print(self.values[idx])
                return self.values[idx]
                        
            min_v = 1000
            
            # determine child indices
            if level == 0: children = [0,1,2]
            else: children = [idx*2, idx*2 + 1]
            
            for c in children:
                cv = max_valuation(idx=c, level=level+1, a=a, b=b)
                min_v = min(cv, min_v)
                
                b = min(b, min_v)
                
                if min_v < a:
                    print('PRUNED!')
                    return min_v
                
            return min_v
        
        def max_valuation(idx, level, a, b):
            if level == 3: return self.values[i]
            
            if level == 2:
                print('\nIn Level 2')
                print('-- Evaluating:', idx, a, b)
            
            max_v = -1000
            
            # determine child indices
            if level == 0: children = [0,1,2]
            else: children = [idx*2, idx*2 + 1]
            
            for c in children:
                cv = min_valuation(idx=c, level=level+1, a=a, b=b)
                
                print(f'-- cv is {cv}')
                
                max_v = min(cv, max_v)
                
                a = min(a, max_v)
                
                if max_v > b:
                    print('PRUNED!')
                    return max_v
                
            return max_v
        
        
        #if mode=='max': max_valuation(0, 0, -1000, 1000)
        #else: min_valuation(0, 0, -1000, 1000)
        
        
        def alpha_beta(idx, level, a, b):
            
            if mode == 'max':
                layer_mode = 'max' if level % 2 == 0 else 'min'
            else:
                layer_mode = 'min' if level % 2 == 0 else 'max'
            
            if level == 3: return self.values[i]
            
            
            print(f'\nIn Level {level}')
            print('-- Evaluating:', idx, a, b)
            
            max_v = -1000
            min_v = 1000
            
            # determine child indices
            if level == 0: children = [0,1,2]
            else: children = [idx*2, idx*2 + 1]
            
            for c in children:
                cv = alpha_beta(idx=c, level=level+1, a=a, b=b)
                
                print(f'-- cv is {cv}')
                
                if layer_mode == 'max':
                    max_v = min(cv, max_v)
                    a = min(a, max_v)
                    if max_v > b:
                        print('PRUNED!')
                        return max_v
                else:
                    min_v = min(cv, min_v)    
                    b = min(b, min_v)
                    if min_v < a:
                        print('PRUNED!')
                        return min_v
                
            return max_v
        
        
        alpha_beta(0, 0, -1000, 1000)
        
        
        

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
        
        print(L0.rstrip())
        print(C01.rstrip())
        print(L1.rstrip())
        print(C12.rstrip())
        print(L2.rstrip())
        print(C23.rstrip())
        print(L3.rstrip())


if __name__ == '__main__':
    
    mm = abstract_minimax()
    #mm.print_values()