"""
This module provides functions for the initial, greedy route construction algorithm

Functions:
 - route_generation(G: nx.Graph, sp_model: ShortestPaths, DEPOT: int) -> tuple[list[list[RouteStep]], list[list[RouteStep]]]:
    Generates a full set of routes given a street network.
"""
import networkx as nx
import math
from shortest_paths import ShortestPaths
import random
import numpy as np
from routes_representations import RouteStep
from costs import single_edge_cost
from collections import deque
from turns import angle_between_vectors, turn_direction
from params import SALT_CAP, ALPHA, SELECTION_WEIGHTS, RAND_THRESH

edges_serviced = 0
def visit_arc(G: nx.Graph, arc: tuple[int, int, int], route: list[RouteStep], options: bool, curr_salt : float, route_required : list[tuple[int, int, int]], all_prev_routes_required: list[list[RouteStep]], undirected=False) -> tuple[int, int]:
    
    """
    Visits an arc and updates the current route and graph information. 
    Returns the new node that the route is currently on and the updated salt value.
    
    Parameters:
        G (networkx.Graph): The graph representing the network
        arc (tuple[int, int, int]): The arc to be visited
        route (list[RouteStep]): The partial route taken so far, represented as a list of RouteStep objects
        options (bool): Flag indicating whether there were other directions for the route to take.
        curr_salt (int): The current value of salt at this point of the route
        route_required (list[tuple[int, int, int]]): List of all required edges in the route so far
        all_prev_routes_required (list): List of all previous routes so far with only required edges
        undirected (bool): Flag indicating if the graph is undirected (default: False)
    Returns:
        int: The new node that the arc is on
        int: The new updated value of salt
    """
    global edges_serviced
    
    from_node = arc[0]
    to_node = arc[1]
    id = arc[2]
    # initialize routstep object
    
    prev_edge = route_required[-1] if len(route_required) > 0 else (all_prev_routes_required[-1][-1] if len(all_prev_routes_required) > 0 else None)
    prev_routestep = RouteStep(prev_edge[0], prev_edge[1], prev_edge[2]) if prev_edge != None else None
    new_routestep = RouteStep(from_node, to_node, id, options=options, saltval=curr_salt)

    # need a check to see if we can actually service the arc given the amount of salt we have left.
    salt_required = G[from_node][to_node][id]['salt_per']
    serviced = G[from_node][to_node][id]['serviced']
    # needs servicing and we can service it. Update weighted and effective degree
    if not serviced and curr_salt >= salt_required:
        G.nodes[from_node]['weighted_degree'] -= G[from_node][to_node][id]['priority']
        
        if (undirected):
            G.nodes[to_node]['weighted_degree'] -= G[from_node][to_node][id]['priority']

        curr_salt -= salt_required

        G[from_node][to_node][id]['passes_rem'] -= 1 # mark that we traversed this arc
        G[from_node][to_node][id]['serviced'] = True
        
        edges_serviced += 1
        new_routestep.deadheaded = False
        route_required.append((from_node, to_node, id))

        # update links
        new_routestep.prev = prev_routestep
        if prev_routestep != None:
            prev_routestep.next = new_routestep
    else:
        G[from_node][to_node][id]['deadheading_passes'] += 1
        new_routestep.deadheaded = True
    
    # if len(route) != 0:
    #     if route[-1].node1 == new_routestep.node1 and route[-1].node2 == new_routestep.node2 and route[-1].edge_id == new_routestep.edge_id:
    #         print("Retraversing same edge immediately")
    #         print("Current route:")
    #         for step in route:
    #             print(step)
    #         print("New arc to visit:", arc)

    route.append(new_routestep)
    return to_node, curr_salt



def get_required_edges_from_node(G: nx.Graph, prev: int, curr: int) -> tuple[list[tuple[int, int, int, dict]], int, int]:
    """
    Identifies all required edges leaving a node.

    Args:
        G (networkX graph): The graph representing the network
        prev (int): The previous node in the route
        curr (int): The current node in the route

    Returns:
        list: the required edges that have the current node as the source
        int: c_min, the smallest cost out of all required edges
        int: c_mzx, the largest cost out of all required edges
    """
    c_min = math.inf
    c_max = -math.inf
    required = []
    for edge in G.edges([curr], data=True, keys=True):
        nxt = edge[1]
        k = edge[2] # identifier for parallel edges
        if edge[3]["serviced"] == False:
            required.append(edge)
            c_min = min(single_edge_cost(G, prev, curr, nxt, k), c_min)
            c_max = max(single_edge_cost(G, prev, curr, nxt, k), c_max)
    return required, c_min, c_max



def return_to_depot(G: nx.Graph, DEPOT: int, route_up_to_now: list[RouteStep], route_required: list[tuple[int, int, int]], salt: int, shortest_paths_model: ShortestPaths, all_prev_routes_required: list[list[RouteStep]], options=False) -> int:
    """
    Returns to the depot via the shortest path, updating any required arcs along the way. 

    Args:
        G (nx.Graph): The graph representing the current network
        DEPOT: The depot of this graph/network
        route_up_to_now (list[RouteStep]): A list of the full current route with deadheading included
        route_required (list[RouteStep]): A list of the full current route without deadheading
        salt (int): Current value of salt
        shortest_paths_model (ShortestPaths): The shortest paths model to use for finding the shortest path
        all_prev_routes_required (list): List of all previous routes so far
        options (bool, optional): Whether the route had options at this point. Defaults to False.
    
    Returns:
        int: The depot node
    """
    if len(route_up_to_now) == 0:
        Exception("Already at depot. This should never happen")

    last_step = route_up_to_now[-1]
    lastEdge = last_step.get_edge()
    path = shortest_paths_model.get_shortest_path(lastEdge, (DEPOT,DEPOT,0))[:-1]

    # slice at 1 to avoid repeating the last edge
    for edge in path[1:]:
        curr, salt = visit_arc(G, edge, route=route_up_to_now, route_required=route_required, options=options, curr_salt=salt, all_prev_routes_required=all_prev_routes_required)
    return curr

def has_edge_within_capacity(G: nx.Graph, node: int, curr_salt: int) -> bool:
    """
    Determines whether the current node has a required edge leading off it within the capacity restrictions
    of the vehicle.

    Args:
        G (nx.Graph): graph representing the network
        node (int): current node that is being examined
        curr_salt (int): current amount of salt being carried

    Returns:
        bool: True if there are edges that can be serviced, False otherwise
    """
    if curr_salt == 0:
        return False
    for neighbor_edge in G.edges(node, data=True, keys=True):
        if neighbor_edge[3]['salt_per'] <= curr_salt and neighbor_edge[3]['serviced'] == False:
            return True
    return False

def find_nearest_required(G : nx.Graph, source: int, salt: int) -> None | list[tuple[int, int, int]]:
    """
    Implements a BFS to find the closest node that has a required edge the vehicle can currently service.

    Args:
        G (nx.Graph): graph representing the network
        source (int): starting node
        salt (int): current amount of salt being carried

    Returns:
        None: if there are no required edges in the graph that are servicable
        list[tuple[int, int, int]]: list of edges to traverse to get to the nearest required edge
    """
    queue = deque()
 
    # Mark the current node as visited and enqueue it
    visited = set()
    queue.append(source)
    parent = dict() # store predecessors to get the nodepath

    # Iterate over the queue
    while queue:
        # Dequeue a vertex from queue
        current = queue.popleft()

        # Get all adjacent vertices of the dequeued vertex currentNode
        # If an adjacent has not been visited, then mark it visited and enqueue it
        for neighbor in G.neighbors(current):
            if neighbor not in visited:
                # new node. Check if it has edge
                visited.add(neighbor)
                queue.append(neighbor)

                if neighbor not in parent:
                    parent[neighbor] = current

                if has_edge_within_capacity(G, neighbor, salt):
                    # get the node path
                    node_path_reversed = list()
                    curr_node = neighbor
                    while curr_node != source:
                        node_path_reversed.append(curr_node)
                        curr_node = parent[curr_node]
                    node_path_reversed.append(source)
                    return reversed_nodes_to_edges(G, node_path_reversed)
    return None

def reversed_nodes_to_edges(G: nx.MultiDiGraph, node_path: list[int]) -> list[tuple[int, int, int]]:
    """
    Takes a sequences of nodes, reverses them, and returns the shortest path traversing
    that list of reversed nodes. Helper function for find_nearest_required

    Args:
        G (nx.MultiDiGraph): graph representing the network
        node_path (list[int]): list of nodes in the path

    Returns:
        list[tuple[int, int, int]]: list of shortest edges to complete a given node path
    """
    edge_path = list()
    for i in range(len(node_path)-1,0,-1):
        node1 = node_path[i]
        node2 = node_path[i-1]
        # in case multiple options, pick the edge with lowest weight
        min_time = math.inf
        min_edge_key = None
        for edge_key, attr in G[node1][node2].items():
            if min_edge_key == None or attr['travel_time'] < min_time:
                min_edge_key = edge_key
                min_time = attr['travel_time']
        edge_path.append((node1, node2, min_edge_key))

    return edge_path


def choose_arc(G: nx.Graph, rcl: list[tuple[int, int, dict]], prev_node: int, weights: list[float, float, float], random_threshold: float) -> tuple[int, int, dict]:
    """
    Selects an arc from a Restricted Candidate List (RCL) based on various weights.

    Parameters:
    - G: The graph representing the road network.
    - rcl: The Restricted Candidate List (RCL) containing the arcs to choose from.
    - prev_node: The previous node in the path.
    - weights: A list of weights used to calculate the arc selection probabilities. The first weight is for turn direction, the second weight is for the degree of the next node, and the third weight is for the priority of the arc.
    - random_threshold: a float between 0 and 1 that determines the probability of choosing an arc randomly.
    Precondition: sum of weights is 1.

    Returns:
    - tuple[int,int,dict]:
        The selected arc from the RCL.

    Algorithm:
    1. If the previous node is None or a random number is greater than 0.8, choose an arc randomly from the RCL.
    2. Otherwise, calculate weights for each arc in the RCL based on turn direction, degree of the next node, and priority.
    3. Normalize the sum of the weights.
    4. Choose an arc randomly based on the weights, where higher weights are more likely to be chosen.

    """

    assert sum(weights) == 1

    # randomize 20% of the time
    if (random.random() < random_threshold):
        return random.choice(rcl)
    
    turn_weights = {"straight": 6, "right": 5, "left": 4, "sharp right": 3, "sharp left": 2, "u-turn": 1}
    weights_turns = np.empty(len(rcl))
    weights_degrees = np.empty(len(rcl))
    weights_priority = np.empty(len(rcl))

    # calculate weights by turn direction, degree of next node, and priority
    i = 0
    for edge in rcl:
        curr_node = edge[0]
        next_node = edge[1]
        k = edge[2]
        # only calculate turn direction if there is a previous node
        if prev_node is not None:
            v_x = G.nodes[curr_node]['x']-G.nodes[prev_node]['x']
            v_y = G.nodes[curr_node]['y']-G.nodes[prev_node]['y']

            w_x = G.nodes[next_node]['x']-G.nodes[curr_node]['x']
            w_y = G.nodes[next_node]['y']-G.nodes[curr_node]['y']

            v = (v_x, v_y)
            w = (w_x, w_y)

            theta = angle_between_vectors(v,w)

            weights_turns[i] = (turn_weights[turn_direction(theta)])
        weights_degrees[i] = G.nodes[next_node]['weighted_degree']
        weights_priority[i] = G[curr_node][next_node][k]['priority']
        i+=1

    # normalize the sum of the weights

    # check for division by zero. need new normalizing. This can occur if there are no required edges leaving a node
    if np.sum(weights_degrees) != 0:
        weights_degrees = weights_degrees / np.sum(weights_degrees)
    if np.sum(weights_priority) != 0:
        weights_priority = weights_priority / np.sum(weights_priority)

    # check that we can calculate turn cost
    if prev_node is not None:
        weights_turns = weights_turns / np.sum(weights_turns)
        weights_tot = weights[0]*weights_turns + weights[1]*weights_degrees + weights[2]*weights_priority
    
    # if there is no previous node, we are at depot, so turn direction doesn't matter.
    else:
        weights_tot = weights[1]*weights_degrees + weights[2]*weights_priority

    # check if weights are all zero (which could occur if at depot and no required edges leaving the node)
    if np.sum(weights_tot) == 0:
        index = int(np.random.choice(np.linspace(0,len(rcl)-1,len(rcl))))
        return rcl[index]

    # normalize the weights again
    weights_tot = weights_tot / np.sum(weights_tot)
    
    # choose an arc based on the weights (higher weights are more likely to be chosen)
    index = int(np.random.choice(np.linspace(0,len(rcl)-1,len(rcl)), p=weights_tot))

    return rcl[index]

def RCA(G: nx.Graph, curr_node: int, route: list[RouteStep], route_required: list[tuple[int, int, int]], DEPOT: int, curr_salt: float, sp_model: ShortestPaths, all_prev_routes_required: list[list[RouteStep]]) -> tuple[list[RouteStep], list[RouteStep]]:
    """
    Implements the Route Construction Algorithm (RCA) for snowplow routing.

    Args:
        G (networkx.Graph): The graph representing the road network.
        curr_node (int): The current node in the route.
        route (list[RouteStep]): The current route.
        route_required (list[tuple[int, int, int]]): The list of already serviced edges in the current route.
        DEPOT (int): The depot node.
        curr_salt (float): The current salt level.
        sp_model (ShortestPaths): The shortest paths model to use for finding the shortest path.
        all_prev_routes_required (list): List of all previous routes so far
    Returns:
        tuple: A tuple containing the complete final route and the set of non-deadheaded arcs in the route.
    """
    
    while True:
        prev_node = route[-1].node1 if len(route) > 0 else None

        required_arcs, c_min, c_max = get_required_edges_from_node(G, prev_node, curr_node)
        rcl = [] # initialize restricted candidate list
        
        for edge in required_arcs:
            cost = single_edge_cost(G, prev_node, edge[0], edge[1], edge[2])
            salt_req = G[edge[0]][edge[1]][edge[2]]['salt_per']

            if cost >= c_min and cost <= c_min + ALPHA*(c_max-c_min) and salt_req <= curr_salt:
                rcl.append(edge)
        
        # chooes an arc based on restricted candidate list if not empty
        if len(rcl) > 0:
            multiple_neighbors = len(G.edges(curr_node)) > 1
            chosen_arc = choose_arc(G, rcl, prev_node, SELECTION_WEIGHTS, RAND_THRESH)
            curr_node, curr_salt = visit_arc(G, chosen_arc, options=multiple_neighbors, route=route, route_required=route_required, curr_salt=curr_salt, all_prev_routes_required=all_prev_routes_required)

        # if restricted candidate list is empty, follow path to nearest node with required arc
        else:
            path = find_nearest_required(G, curr_node, curr_salt)
            # no more required arcs in the graph that we can service, so we're done.
            # return to the depot and refill salt cap
            if path is None:
                curr_node = return_to_depot(G, DEPOT, route, route_required, curr_salt, sp_model, options=False, all_prev_routes_required=all_prev_routes_required)
                return route, route_required
            
            # otherwise go to the arc to visit
            for edge in path:
                multiple_neighbors = len(G.edges(curr_node)) > 1 # could we take a different path
                curr_node, curr_salt = visit_arc(G, edge, options=multiple_neighbors, route=route, route_required=route_required, curr_salt=curr_salt, all_prev_routes_required=all_prev_routes_required)
                if curr_node == DEPOT:
                    return route, route_required
        # if we're at the depot for any reason, that's the end of a route.
        if curr_node == DEPOT:
            return route, route_required
        
def all_serviced(total_required):
    global edges_serviced
    return edges_serviced == total_required

def calc_total_required_edges(G: nx.Graph) -> int:
    """
    Calculates the total number of required edges in the graph.

    Args:
        G (nx.Graph): The graph representing the street network.

    Returns:
        int: The total number of required edges in the graph.
    """
    total_required = 0
    for edge in G.edges(data=True, keys=True):
        if edge[3]['serviced'] == False:
            total_required += 1
    return total_required

def route_generation(G: nx.Graph, sp_model: ShortestPaths, DEPOT: int) -> tuple[list[list[RouteStep]], list[list[tuple[int, int, int]]]]:
    """
    Generates a full set of routes given a street network.
    Makes a copy of the original graph to avoid modifying the original.

    Args:
        G (nx.Graph): the graph representing the street network
        sp_model (ShortestPaths): the shortest paths model to use for finding the shortest path
        DEPOT (int): The depot node
    Returns:
        tuple[list[list[RouteStep]], list[list[RouteStep]]]: Two lists. 
        The first contains a continuous set of routes, while the second contains only the required edges of the route (so no deadheading edges are included).
    """
    global edges_serviced
    G_copy = G.copy()
    curr_salt = SALT_CAP
    curr_node = DEPOT
    edges_serviced = 0
    total_required = calc_total_required_edges(G_copy)

    routes = []
    routes_only_required = []

    partial_route: list[RouteStep] = list()
    partial_route_required: list[tuple[int, int, int]] = list()
    while all_serviced(total_required) == False:
        partial_route, partial_route_required = RCA(G_copy, curr_node, partial_route, partial_route_required, DEPOT, curr_salt, sp_model, routes_only_required)
        
        routes.append(partial_route)
        routes_only_required.append(partial_route_required)
        
        curr_salt = SALT_CAP
        curr_node = DEPOT

        partial_route = list()
        partial_route_required = list()
    
    return routes, routes_only_required

if __name__ == "__main__":
    from main import G, shortest_paths
    import plotting
    r, rreq = route_generation(G, shortest_paths)
    print("Printing route")
    for route in rreq:
        for edge in route:
            print(edge)

    G_graph = plotting.add_order_attribute(G, rreq)
    plotting.draw_labeled_multigraph(G_graph, 'order')