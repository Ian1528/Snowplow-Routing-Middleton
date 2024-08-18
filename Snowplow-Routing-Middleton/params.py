import math
import networkx as nx
# graph construction and vehicle caps
DEPOT = 0
SALT_CAP = 20000

# route construction
ALPHA = 1
SELECTION_WEIGHTS = [0.4, 0.2, 0.4]
RAND_THRESH = 0.2

# cost function
COST_WEIGHTS = [.4, .2, .4]

# local search
K = 3

# genetic algorithm
POP_SIZE = 10
N_ITER = 25
BETA = .7

parameters = [DEPOT, SALT_CAP, ALPHA, SELECTION_WEIGHTS]

def set_params(params: dict) -> None:
    """
    Writes the parameters to the params file.

    Args:
        params (dict): dictionary of parameter values.
    """
    names = {}
    # if params == None:
    
    # else:



def read_params():
    """
    
    """

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
    return minNode[0], minNode[1]