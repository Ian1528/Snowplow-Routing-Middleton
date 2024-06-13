import networkx as nx
from turns import angle_between_vectors, turn_direction

def single_edge_cost(G: nx.Graph, prev: int, curr: int, nxt: int, k: int) -> float:
    """
    Returns the cost of traversing an edge between two nodes. 
    Cost is based on travel time and turn direction.

    Args:
        G (nx.Graph): The graph representing the street network
        prev (int): Previous node of the route. None if there is no previous node
        curr (int): Start node of the edge
        nxt (int): End node of the edge
        k (int): ID of the edge (in a multigraph)

    Returns:
        float: The cost associated with traveling on the dge
    """

    cost = G[curr][nxt][k]['travel_time']

    # without previous node, we can't factor turn direction
    if prev is None:
        return cost
    
    # with a previous node, we incorporate turning penalites
    turn_cost = {"straight": 0, "right": 1, "left": 2, "sharp right": 2, "sharp left": 3, "u-turn": 4}

    v_x = G.nodes[curr]['x']-G.nodes[prev]['x']
    v_y = G.nodes[curr]['y']-G.nodes[prev]['y']

    w_x = G.nodes[nxt]['x']-G.nodes[curr]['x']
    w_y = G.nodes[nxt]['y']-G.nodes[curr]['y']

    v = (v_x, v_y)
    w = (w_x, w_y)

    theta = angle_between_vectors(v,w)
    cost += turn_cost[turn_direction(theta)]

    return cost

def cost_of_dual_node(first_edge: tuple[int, int, int, dict], angle: float) -> float:
    """
    Calculate the weighted degree of a node in a graph. Helper for ``create_dual``.
    
    Parameters:
        first_edge (tuple[int, int, int, dict]): The dual node, which is an edge in the primal graph.
        angle (float): The angle of the turn with the next edge.
        
    Returns:
        float: The weighted degree of the node.
        
    Notes:
        The nodes passed as parameters are edges in the original primal graph.
    """
    weight = first_edge[3]['travel_time']
    # add the turn penalty cost
    turn_penalty = {"straight": 0, "right": 1, "left": 2, "sharp right": 2, "sharp left": 3, "u-turn": 4}
    
    weight += turn_penalty[turn_direction(angle)]
    return weight
