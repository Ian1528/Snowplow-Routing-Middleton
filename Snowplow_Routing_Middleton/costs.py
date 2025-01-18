"""
This module provides functions to calculate various costs associated with snowplow routing on a street network graph.

Functions:
    route_travel_time(G: nx.MultiDiGraph, route: list[tuple[int, int, int]], DEPOT: int) -> float:
        Calculate the total travel time of a route.
    single_edge_cost(G: nx.Graph, prev: int, curr: int, nxt: int, k: int) -> float:
        Returns the cost of traversing an edge between two nodes given the previous node.
    cost_of_dual_node(first_edge: tuple[int, int, int, dict], angle: float) -> float:
        Calculate the cost of visiting a node in the dual graph based on turn directions and travel times.
    routes_cost_linked_list(G: nx.MultiDiGraph, shortest_paths: ShortestPaths, head, DEPOT: int) -> float:
        Calculates the total cost of a route represented as a linked list of edges.
    routes_cost(G: nx.Graph, shortest_paths: ShortestPaths, routes: list[list[tuple[int, int, int]]], DEPOT: int) -> float:
        Calculates the total cost of a full set of routes, represented as a 2D list of route step objects.
"""

import networkx as nx
from .turns import angle_between_vectors, turn_direction
from .shortest_paths import ShortestPaths
from .params import COST_WEIGHTS, TURN_WEIGHT, PRIORITY_SCALE_FACTOR, SALT_CAP

def route_travel_time(G: nx.MultiDiGraph, route: list[tuple[int, int, int]], DEPOT: int) -> float:
    """
    Calculate the total travel time of a route.
    
    Parameters:
        G (nx.MultiDiGraph): The graph representing the street network.
        route (list[tuple[int, int, int]]: The route to be evaluated.
        DEPOT (int): The depot node.
    Returns:
        float: The total travel time of the route.
    """
    travel_time = 0
    for edge in route:
        if edge == (DEPOT, DEPOT, 0):
            continue
        if G.get_edge_data(edge[0], edge[1], edge[2]) == None:
            print("no edge,", edge, DEPOT)
            continue
        travel_time += G.get_edge_data(edge[0], edge[1], edge[2])['travel_time']
    return travel_time
def single_edge_cost(G: nx.Graph, prev: int, curr: int, nxt: int, k: int) -> float:
    """
    Returns the cost of traversing an edge between two nodes given the previous node.
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
    cost += TURN_WEIGHT*turn_cost[turn_direction(theta)]

    return cost

def cost_of_dual_node(first_edge: tuple[int, int, int, dict], angle: float) -> float:
    """
    Calculate the cost of visiting a node in the dual graph based on turn directions and travel times.
    Helper for ``create_dual``.
    
    Parameters:
        first_edge (tuple[int, int, int, dict]): The dual node, which is an edge in the primal graph.
        angle (float): The angle of the turn with the next edge.
        
    Returns:
        float: The weighted degree of the node.
        
    Notes:
        The nodes passed as parameters are edges in the original primal graph.
    """
    weight = first_edge[3]['travel_time']

    # add the turn penalty cost, estimated extra time to make the turn
    turn_penalty = {"straight": 0, "right": 5, "left": 10, "sharp right": 10, "sharp left": 15, "u-turn": 120}
    
    weight += TURN_WEIGHT * turn_penalty[turn_direction(angle)]
    return weight

def routes_cost_linked_list(G: nx.MultiDiGraph, shortest_paths: ShortestPaths, head, DEPOT: int) -> float:
    """
    Calculates the total cost of a route represented as a linked list of edges.
    Parameters:
        G (nx.MultiDiGraph): The graph representing the network.
        shortest_paths (ShortestPaths): The object containing the shortest paths information.
        head (Node): The head node of the linked list.
        DEPOT (int): The depot node.
    Returns:
        float: The total cost of the route.
    """
    
    time_cost = 0
    priority_cost = 0
    deadhead_cost = 0
    time = 0

    salt_val = SALT_CAP
    deadheading_time = 0
    node = head.next
    while node.next != None:
        edge = node.data
        next_edge = node.next.data
        edge_data = G[edge[0]][edge[1]][edge[2]]

        if salt_val - edge_data['salt_per'] < 0 or node.is_route_end:
            # deadheading stuff on the first trip back
            path_to_depot = shortest_paths.get_shortest_path(edge, (DEPOT,DEPOT,0))
            if len(path_to_depot) > 2:
                deadheading_time += shortest_paths.get_dist(path_to_depot[1], (DEPOT,DEPOT,0))

            # time stuff on the first trip back
            time_cost += shortest_paths.get_dist(edge, (DEPOT,DEPOT,0))
            
            # update salt
            salt_val = SALT_CAP

            # deadheading stuff on the trip out
            path_to_next_edge = shortest_paths.get_shortest_path((DEPOT,DEPOT,0), next_edge)
            if len(path_to_next_edge) > 2:
                deadheading_time += shortest_paths.get_dist(path_to_next_edge[1], next_edge)
            # time stuff on the trip out
            time_cost += shortest_paths.get_dist((DEPOT,DEPOT,0), next_edge)
        else:
            # deadheading time
            if edge[1] != next_edge[0]:
                path = shortest_paths.get_shortest_path(edge, next_edge)
                deadheading_time += shortest_paths.get_dist(path[1], next_edge)
            # normal time stuff
            time_cost += shortest_paths.get_dist(edge, next_edge)
        # penalize priorities
        time += edge_data['travel_time']
        priority_cost += (edge_data['priority'] * time * PRIORITY_SCALE_FACTOR)
        salt_val -= edge_data['salt_per']
        node = node.next            
    # penalize deadheading
    deadhead_cost = deadheading_time

    all_costs = [time_cost, deadhead_cost, priority_cost]
    # print("linkedListcost, [time, deadhead, priority]", all_costs)
    total_cost = 0
    for i in range(3):
        total_cost += all_costs[i]*COST_WEIGHTS[i]
    return total_cost
def routes_cost(G: nx.Graph, shortest_paths: ShortestPaths, routes: list[list[tuple[int, int, int]]], DEPOT: int) -> float:
    """
    Calculates the total cost of a full set of routes, represented as a 2d list of routestep objects.

    Args:
        G (nx.Graph): the graph of the network
        shortest_paths (ShortestPaths): the shortestpaths object related to that graph (used for getting costs)
        routes (list[list[RouteStep]]): the routes to be evaluated
        DEPOT (int): the depot node

    Returns:
        float: the cost of the route
    """
    time_cost = 0
    priority_cost = 0
    deadhead_cost = 0
    time = 0

    salt_val = SALT_CAP
    deadheading_time = 0
    for i in range(len(routes)):
        try:
            route = routes[i]
        except:
            print("route", i, "is", routes)
            raise Exception("route is not a list")
        for j in range(len(route)):
            edge = route[j]
            edge_data = G[edge[0]][edge[1]][edge[2]]

            next_edge = route[j+1] if j+1 < len(route) else routes[i+1][0] if i+1 < len(routes) else None

            if salt_val - edge_data['salt_per'] < 0 or j == len(route)-1:
                # deadheading stuff
                path_to_depot = shortest_paths.get_shortest_path(edge, (DEPOT,DEPOT,0))
                if len(path_to_depot) > 2:
                    deadheading_time += shortest_paths.get_dist(path_to_depot[1], (DEPOT,DEPOT,0))

                # normal time stuff
                time_cost += shortest_paths.get_dist(edge, (DEPOT,DEPOT,0))
                salt_val = SALT_CAP
                if next_edge is not None:
                    # deadheading stuff
                    path_to_next_edge = shortest_paths.get_shortest_path((DEPOT,DEPOT,0), next_edge)
                    if len(path_to_next_edge) > 2:
                        deadheading_time += shortest_paths.get_dist(path_to_next_edge[1], next_edge)

                    # normal time stuff
                    time_cost += shortest_paths.get_dist((DEPOT,DEPOT,0), next_edge)
            else:
                # deadheading time
                if edge[1] != next_edge[0]:
                    path = shortest_paths.get_shortest_path(edge, next_edge)
                    deadheading_time += shortest_paths.get_dist(path[1], next_edge)
                # normal time stuff
                time_cost += shortest_paths.get_dist(edge, next_edge)
                
            # penalize priorities
            time += edge_data['travel_time']
            priority_cost += (edge_data['priority'] * time * PRIORITY_SCALE_FACTOR)
            salt_val -= edge_data['salt_per']
    # penalize deadheading
    deadhead_cost = deadheading_time

    all_costs = [time_cost, deadhead_cost, priority_cost]
    # print("[time, deadhead, priority]", all_costs)
    total_cost = 0
    for i in range(3):
        total_cost += all_costs[i]*COST_WEIGHTS[i]
    return total_cost

def calculate_time_deadheading(routes: list[list[tuple[int, int, int]]], shortest_paths: ShortestPaths, DEPOT: int) -> float:
    """
    Calculate the total time spent deadheading in a set of routes.

    Args:
        routes (list[list[tuple[int, int, int]]): The routes to be evaluated.
        shortest_paths (ShortestPaths): The object containing the shortest paths information.
        DEPOT (int): The depot node.

    Returns:
        float: The total time spent deadheading.
    """
    deadheading_time = 0
    for i in range(len(routes)):
        route = routes[i]
        for j in range(len(route)-1):
            edge = route[j]
            next_edge = route[j+1]
            if edge[1] != next_edge[0]:
                path = shortest_paths.get_shortest_path(edge, next_edge)
                deadheading_time += shortest_paths.get_dist(path[1], next_edge)
        path_to_depot = shortest_paths.get_shortest_path(route[-1], (DEPOT, DEPOT, 0))

        # sometimes, the last ending to the depot isn't a deadhead at all, but is being serviced
        if len(path_to_depot) > 2:
            deadheading_time += shortest_paths.get_dist(path_to_depot[1], (DEPOT, DEPOT, 0))
        
    return deadheading_time