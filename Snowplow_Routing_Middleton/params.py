"""
This module defines various parameters and a function for finding the depot node in a graph.

Constants:
- DEPOT (int): The ID of the depot node.
- SALT_CAP (int): The salt capacity of the vehicle.
- PLOW_SPEED_RESIDENTIAL (float): The plowing speed in residential areas (m/s).
- PLOW_SPEED_HIGHWAY (float): The plowing speed on highways (m/s).
- ALPHA (int): A parameter for route construction.
- SELECTION_WEIGHTS (list[float]): Weights for selection criteria.
- RAND_THRESH (float): Threshold for random selection.
- COST_WEIGHTS (list[float]): Weights for cost function components.
- TURN_WEIGHT (float): Weight for turns in the cost function.
- PRIORITY_SCALE_FACTOR (float): Scale factor for priority in the cost function.
- K (int): Parameter for local search.
- KAPPA (int): Number of nearest neighbors to consider when inserting an edge in crossover algorithm.
- POP_SIZE (int): Population size for the genetic algorithm.
- N_ITER (int): Number of iterations for the genetic algorithm.
- BETA (float): Parameter for the genetic algorithm.

Functions:
- find_depot(G: nx.MultiDiGraph) -> tuple[int, dict]:

"""
import math
import networkx as nx
# graph construction and vehicle caps
DEPOT = 0

SALT_CAP = 20000
PLOW_SPEED_RESIDENTIAL = 4.91744 # m/s
PLOW_SPEED_HIGHWAY = 8.9408 # m/s
HIGH_PRIORITY_ROADS = ["Parmenter Street", "Airport Road", "Pleasant View Road", "Deming Way", "University Avenue", "Pheasant Branch Road", "Park Street", "North Gammon Road", "North High Point Road", "High Road"]
# notes on high priority roads:
#  - century avenue west of -89.509817
#  - park st south of 43.097088


# route construction
ALPHA = 1
SELECTION_WEIGHTS = [0.4, 0.2, 0.4]
RAND_THRESH = 0.2

# cost function
COST_WEIGHTS = [.8, .1, .1]
TURN_WEIGHT = 0.01
PRIORITY_SCALE_FACTOR = .0005
# local search
K = 3

# crossover
KAPPA = 3 # number of nearest neighbors to consider when inserting edge

# genetic algorithm
POP_SIZE = 10
N_ITER = 25
BETA = .7

# parameters = [DEPOT, SALT_CAP, ALPHA, SELECTION_WEIGHTS]

def find_depot(G: nx.MultiDiGraph) -> tuple[int, dict]:
    """
    Finds the depot node in a given graph based on the Euclidean distance from a fixed point.
    Parameters:
    - G (nx.MultiDiGraph): The graph to search for the depot node.
    Returns:
    - tuple[int, dict]: A tuple containing the ID and attributes of the depot node.
    """
    depX = -89.513456
    depY = 43.123172

    dist = lambda x,y : math.sqrt((x-depX)**2 + (y-depY)**2)

    minDist = math.inf
    minNode = None
    for node in G.nodes(data=True):
        attr = node[1]
        distance = dist(attr['x'], attr['y'])
        if distance < minDist:
            minDist = distance
            minNode = node
    return minNode[0], minNode[1]['x'], minNode[1]['y']