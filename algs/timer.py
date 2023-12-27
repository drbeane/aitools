class Timer:
    
    def __init__(self):
        self.sections = {}
        self.section_starts = {}
        self.section_counts = {}
        
    def start(self, section):
        from timeit import default_timer as timer
        
        start = self.section_starts.get(section, None)
        if start is not None:
            print(f'Timer is already running for section [{section}].')
        else:
            self.section_starts[section] = timer()
            
    def stop(self, section):
        from timeit import default_timer as timer

        start = self.section_starts.get(section, None)
        if start is None:
            print(f'Timer is not yet running for section [{section}].')
        else:
            self.sections[section] = self.sections.get(section, 0) + timer() - start
            self.section_counts[section] = self.section_counts.get(section, 0) + 1
            self.section_starts[section] = None
    
    def report(self):
        import pandas as pd

        s = list(self.sections.keys())
        t = [self.sections[k] for k in s]
        n = [self.section_counts[k] for k in s]
        
        df = pd.DataFrame(dict(section=s, count=n, time=t))
        df = df.sort_values('time', ascending=False)
        
        print()
        print(df)
        print()
    
        
    