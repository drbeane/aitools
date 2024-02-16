#-------------------------------------
# Init file for environments module
#-------------------------------------

#-----------------------------------------------------------
# Import statements below enable importing environments 
# directly from aitools.envs rather than from submodules
#-----------------------------------------------------------
from aitools.envs.route_planning import RoutePlanning
from aitools.envs.npuzzle import NPuzzle
from aitools.envs.tsp import TSP    
from aitools.envs.connectx import ConnectX
from aitools.envs.oware import Oware
from aitools.envs.gridland import GridLand
from aitools.envs.knapsack import Knapsack
from aitools.envs.job_assignment import JobAssignment
from aitools.envs.frozen_platform import FrozenPlatform