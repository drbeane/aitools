#-------------------------------------
# Init file for environments module
#-------------------------------------

print('loading envs')

#-----------------------------------------------------------
# Import statements below enable importing environments 
# directly from aitools.envs rather than from submodules
#-----------------------------------------------------------
from aitools.envs.route_planning import RoutePlanning
from aitools.envs.npuzzle import NPuzzle
from aitools.envs.tsp import TSP                        