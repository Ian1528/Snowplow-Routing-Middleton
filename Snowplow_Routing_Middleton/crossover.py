"""
This module provides functions for the crossover operation to combine two routes.
Functions:
    apply_crossover(G: nx.MultiDiGraph, sp: ShortestPaths, routes1: list[list[tuple[int, int, int]]], routes2: list[list[tuple[int, int, int]]], DEPOT: int) -> list[list[tuple[int, int, int]]]:
        Takes two full sets of routes and returns the crossover between them.
"""
import math
import numpy as np
import networkx as nx
from .costs import routes_cost
from .shortest_paths import ShortestPaths
from .routes_representations import RouteStep
from .params import KAPPA

def combine(routes1: list[tuple[int, int, int]], routes2: list[tuple[int, int, int]]) -> tuple[list[tuple[int, int, int]], int]:
    """
    Combines two parent routes into a child route

    Args:
        routes1 (list[tuple[int, int, int]]): parent route 1
        routes2 (list[tuple[int, int, int]]): parent route 2

    Returns:
        routes0 (tuple[list[tuple[int, int, int]], int, int]): the new child route, as well as start and endpoints of cutting (end is exclusive)
    """
    start, end = 0, 0
    while abs(start-end) < 5: # could change 5 to be a varying parameter
        start, end = np.random.randint(0, len(routes1)), np.random.randint(0, len(routes1))
    
    if start > end:
        start, end = end, start
    routes_new = routes1[:start] + routes2[start:end] + routes1[end:]
    return routes_new


def remove_duplicates(route: list[tuple[int, int, int]], sp: ShortestPaths, DEPOT: int) -> None:
    """
    Remove all duplicated edges in a set of routes by choosing the one that minimizes distance.
    Modifies the existing list in-place

    Args:
        changed_route_index (int): index of insertion, which may have added duplicates.
        routes (list[list[RouteStep]]): set of current routes
        sp (ShortestPaths): corresponding shortest paths object
        DEPOT (int): depot node
    """
    def include(index: int):
        # get all indices of occurences of the edge
        occurrences = [i for i, edge in enumerate(route) if edge == route[index]]
        if len(occurrences) == 1:
            return True
        i1 = occurrences[0]
        i2 = occurrences[1]
        edge = route[index]

        prev_edge = (DEPOT,DEPOT,0) if i1 == 0 else route[i1-1]
        next_edge = (DEPOT,DEPOT,0) if i1 == len(route)-1 else route[i1+1]
        diff1 = sp.get_dist(prev_edge, edge) + sp.get_dist(edge, next_edge) - sp.get_dist(prev_edge, next_edge)

        prev_edge = (DEPOT,DEPOT,0) if i2 == 0 else route[i2-1]
        next_edge = (DEPOT,DEPOT,0) if i2 == len(route)-1 else route[i2+1]
        diff2 = sp.get_dist(prev_edge, edge) + sp.get_dist(edge, next_edge) - sp.get_dist(prev_edge, next_edge)

        if diff2 > diff1:
            # keep i1, discard i2
            return i1 == index
        else:
            # keep i2, discard i1
            return i2 == index
    route[:] = [edge for i, edge in enumerate(route) if include(i)]

def get_missing_edges(G: nx.MultiDiGraph, routes: list[tuple[int, int, int]]) -> set:
    """
    Given a graph network and a current solution of routes, find all edges that still haven't been serviced.

    Args:
        G (nx.MultiDiGraph): graph representing network
        routes (list[list[RouteStep]]): current routes

    Returns:
        set: set of unserviced edges
    """
    
    required_edges = set(edge[:3] for edge in G.edges(data=True, keys=True) if edge[3]['priority'] != 0)
    missing_edges = required_edges-set(routes)
    return missing_edges

def insert_edge(G: nx.MultiDiGraph, edge: RouteStep, route: list[tuple[int, int, int]], sp: ShortestPaths, N: int, DEPOT: int) -> list[tuple[int, int, int]]:
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
    best_route = None

    neighbors = [tuple(i) for i in sp.nearest_neighbors[edge][1:N+1]]
    for i in range(len(route)+1):
        if i != len(route) and route[i] not in neighbors:
            continue
        new_route = route[:i] + [edge] + route[i:]
        new_cost = routes_cost(G, sp, new_route, DEPOT)
        if new_cost < best_cost:
            best_cost = new_cost
            best_route = new_route
    return best_route

def apply_crossover(G: nx.MultiDiGraph, sp: ShortestPaths, routes1: list[tuple[int, int, int]], routes2: list[tuple[int, int, int]], DEPOT: int) -> list[tuple[int, int, int]]:
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
    routes0 = combine(routes1, routes2)

    remove_duplicates(routes0, sp, DEPOT)

    for edge in get_missing_edges(G, routes0):
        routes0 = insert_edge(G, edge, routes0, sp, KAPPA, DEPOT)
    return routes0