#-------------------------------------
# Init file for environments module
#-------------------------------------

print('loading dev envs')

#-----------------------------------------------------------
# Import statements below enable importing environments 
# directly from aitools.envs rather than from submodules
#-----------------------------------------------------------
from aitools.dev.envs.dev_route_planning import RoutePlanning
from aitools.dev.envs.dev_npuzzle import NPuzzle
from aitools.dev.envs.dev_tsp import TSP                        