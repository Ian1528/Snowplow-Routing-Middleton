# graph construction and vehicle caps
DEPOT = 0
SALT_CAP = 100

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