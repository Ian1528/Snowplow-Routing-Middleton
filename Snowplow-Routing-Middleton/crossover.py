import copy
import math
import numpy as np
import networkx as nx
from costs import routes_cost
from shortest_paths import ShortestPaths
from routes_representations import RouteStep
from params import KAPPA

def combine(routes1: list[list[tuple[int, int, int]]], routes2: list[list[tuple[int, int, int]]]) -> tuple[list[list[tuple[int, int, int]]], int]:
    """
    Combines two parent routes into a child route

    Args:
        routes1 (list[list[RouteStep]]): parent route 1
        routes2 (list[list[RouteStep]]): parent route 2

    Returns:
        tuple[list[list[RouteStep]], int]: the new child route, as well as the index of the route that was replaced
    """
    routes0 = copy.deepcopy(routes1)
    remove_index = int(np.random.random()*len(routes1))
    add_index = int(np.random.random()*len(routes2))

    # make the swap
    try:
        routes0[remove_index] = routes2[add_index].copy()
    except:
        print("Operation not permitted. Routes are the folloiwng lengths: ", len(routes0), len(routes2), remove_index, add_index)
        print(routes0)
        print(routes2)
        raise Exception()
    return routes0, remove_index


def remove_duplicates(changed_route_index: int, routes: list[list[tuple[int, int, int]]], sp: ShortestPaths, DEPOT: int):
    """
    Remove all duplicated edges in a set of routes by choosing the one that minimizes distance.
    Modifies the existing list in-place

    Args:
        changed_route_index (int): index of insertion, which may have added duplicates.
        routes (list[list[RouteStep]]): set of current routes
        sp (ShortestPaths): corresponding shortest paths object
        DEPOT (int): depot node
    """
    added_route = routes[changed_route_index]
    for h in range(len(routes)):
        route = routes[h]
        if h == changed_route_index:
            continue
    
        # loop backwards so deletion while iterating is feasible
        for i in range(len(route)-1, -1, -1):
            edge = route[i]
            try:
                j = added_route.index(edge)
                prev_edge = (DEPOT,DEPOT,0) if j == 0 else added_route[j-1]
                next_edge = (DEPOT,DEPOT,0) if j == len(added_route)-1 else added_route[j+1]

                diff1 = sp.get_dist(prev_edge, edge) + sp.get_dist(edge, next_edge) - sp.get_dist(prev_edge, next_edge)

                prev_edge = (DEPOT,DEPOT,0) if i == 0 else route[i-1]
                next_edge = (DEPOT,DEPOT,0) if i == len(route)-1 else route[i+1]

                diff2 = sp.get_dist(prev_edge, edge) + sp.get_dist(edge, next_edge) - sp.get_dist(prev_edge, next_edge)

                # remove the place with the smaller diff
                if diff2 > diff1:
                    del added_route[j] # del because we know the index for certain
                else:
                    del route[i] # remove b/c indices might change with multiple removals
            except ValueError:
                # no duplicate
                continue

def get_missing_edges(G: nx.MultiDiGraph, routes: list[list[tuple[int, int, int]]]) -> set:
    """
    Given a graph network and a current solution of routes, find all edges that still haven't been serviced.

    Args:
        G (nx.MultiDiGraph): graph representing network
        routes (list[list[RouteStep]]): current routes

    Returns:
        set: set of unserviced edges
    """
    condensed_routes = list()
    for route in routes:
        for edge in route:
            condensed_routes.append(edge)
    
    required_edges = set(edge[:3] for edge in G.edges(data=True, keys=True) if edge[3]['priority'] != 0)

    missing_edges = required_edges-set(condensed_routes)
    return missing_edges


# def insert_edge(G: nx.MultiDiGraph, edge: RouteStep, routes : list[list[tuple[int, int, int]]], sp: ShortestPaths) -> list[list[tuple[int, int, int]]]:
#     """
#     Inserts an edge into the graph greedily. Finds the cost with respect to each insertion point
#     Args:
#         G (nx.MultiDiGraph): graph representing network
#         edge (RouteStep): the edge to be inserted
#         routes (list[list[RouteStep]]): the current set of routes
#         sp (ShortestPaths): shortest paths object corresponding to the graph

#     Returns:
#         list[list[RouteStep]]: the new set of routes with the edge inserted
#     """
#     best_cost = math.inf
#     best_routes = None
#     for route_id in range(len(routes)):
#         route = routes[route_id]

#         for i in range(len(route)+1):
#             new_route = route[:i] + [edge] + route[i:]
#             new_full_routes = routes[:route_id] + [new_route] + routes[route_id+1:]
#             new_cost = routes_cost(G, sp, new_full_routes)
#             if new_cost < best_cost:
#                 best_cost = new_cost
#                 best_routes = new_full_routes

#     return best_routes

def insert_edge(G: nx.MultiDiGraph, edge: RouteStep, routes : list[list[tuple[int, int, int]]], sp: ShortestPaths, N: int, DEPOT: int,) -> list[list[tuple[int, int, int]]]:
    """
    Inserts an edge into the graph greedily. Finds the cost with respect to each insertion point
    Args:
        G (nx.MultiDiGraph): graph representing network
        edge (RouteStep): the edge to be inserted
        routes (list[list[RouteStep]]): the current set of routes
        sp (ShortestPaths): shortest paths object corresponding to the graph
        N (int): number of nearest neighbors to consider
        DEPOT (int): depot node

    Returns:
        list[list[RouteStep]]: the new set of routes with the edge inserted
    """
    best_cost = math.inf
    best_routes = None

    neighbors = [tuple(i) for i in sp.nearest_neighbors[edge][1:N+1]]
    for route_id in range(len(routes)):
        route = routes[route_id]

        for i in range(len(route)+1):
            if i != len(route) and route[i] not in neighbors:
                    continue

            new_route = route[:i] + [edge] + route[i:]
            new_full_routes = routes[:route_id] + [new_route] + routes[route_id+1:]
            new_cost = routes_cost(G, sp, new_full_routes, DEPOT)
            if new_cost < best_cost:
                best_cost = new_cost
                best_routes = new_full_routes

    return best_routes



def apply_crossover(G: nx.MultiDiGraph, sp: ShortestPaths, routes1: list[list[tuple[int, int, int]]], routes2: list[list[tuple[int, int, int]]], DEPOT: int) -> list[list[tuple[int, int, int]]]:
    """
    Takes two full sets of routes and returns the crossover between them

    Args:
        G (nx.MultiDiGraph): the graph of the network
        sp (ShortestPaths): shortest paths object corresponding to the graph network
        routes1 (list[list[RouteStep]]): the first set of routes
        routes2 (list[list[RouteStep]]): the second set of routes
        DEPOT (int): the depot node

    Returns:
        list[list[RouteStep]]: new set of routes
    """
    routes0, change_index = combine(routes1, routes2)

    remove_duplicates(change_index, routes0, sp, DEPOT)

    for edge in get_missing_edges(G, routes0):
        routes0 = insert_edge(G, edge, routes0, sp, KAPPA, DEPOT)
    return routes0