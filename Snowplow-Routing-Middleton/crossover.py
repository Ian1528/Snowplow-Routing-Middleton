import copy
import math
import numpy as np
import networkx as nx
from costs import routes_cost
from shortest_paths import ShortestPaths
from routes_representations import RouteStep
from params import DEPOT

def combine(routes1: list[list[RouteStep]], routes2: list[list[RouteStep]]) -> tuple[list[list[RouteStep]], int]:
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
    routes0[remove_index] = routes2[add_index]
    return routes0, remove_index


def remove_duplicates(changed_route_index: int, routes: list[list[RouteStep]], sp: ShortestPaths):
    """
    Remove all duplicated edges in a set of routes by choosing the one that minimizes distance.
    Modifies the existing list in-place

    Args:
        changed_route_index (int): index of insertion, which may have added duplicates.
        routes (list[list[RouteStep]]): set of current routes
        sp (ShortestPaths): corresponding shortest paths object
    """
    added_route = routes[changed_route_index]

    for h in range(len(routes)):
        route = routes[h]
        new_route = route.copy()
        if h == changed_route_index:
            continue
        for i in range(len(route)):
            step = route[i]
            edge = step.get_edge()

            if step in added_route:
                # duplicate
                j = added_route.index(step)
                prev_edge = (DEPOT,DEPOT,0) if j == 0 else added_route[j-1].get_edge()
                next_edge = (DEPOT,DEPOT,0) if j == len(added_route)-1 else added_route[j+1].get_edge()

                diff1 = sp.get_dist(prev_edge, edge) + sp.get_dist(edge, next_edge) - sp.get_dist(prev_edge, next_edge)

                prev_edge = (DEPOT,DEPOT,0) if i == 0 else route[i-1].get_edge()
                next_edge = (DEPOT,DEPOT,0) if i == len(route)-1 else route[i+1].get_edge()

                diff2 = sp.get_dist(prev_edge, edge) + sp.get_dist(edge, next_edge) - sp.get_dist(prev_edge, next_edge)
                
                # compare the two places to remove the edge
                if diff2 > diff1:
                    added_route.remove(step)
                else:
                    new_route.remove(step)
        route = new_route


def get_missing_edges(G: nx.MultiDiGraph, routes: list[list[RouteStep]]) -> set:
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
        for step in route:
            condensed_routes.append(step.get_edge())
    
    required_edges = set(edge[:3] for edge in G.edges(data=True, keys=True) if edge[3]['priority'] != 0)

    missing_edges = required_edges-set(condensed_routes)
    return missing_edges


def insert_edge(G: nx.MultiDiGraph, edge: RouteStep, routes : list[list[RouteStep]], sp: ShortestPaths) -> list[list[RouteStep]]:
    """
    Inserts an edge into the graph greedily. Finds the cost with respect to each insertion point
    Args:
        G (nx.MultiDiGraph): graph representing network
        edge (RouteStep): the edge to be inserted
        routes (list[list[RouteStep]]): the current set of routes
        sp (ShortestPaths): shortest paths object corresponding to the graph

    Returns:
        list[list[RouteStep]]: the new set of routes with the edge inserted
    """
    best_cost = math.inf
    best_routes = None
    for route_id in range(len(routes)):
        route = routes[route_id]

        for i in range(len(route)+1):
            new_route = route[:i] + [edge] + route[i:]
            new_full_routes = routes[:route_id] + [new_route] + routes[route_id+1:]
            new_cost = routes_cost(G, sp, new_full_routes)
            if new_cost < best_cost:
                best_cost = new_cost
                best_routes = new_full_routes

    return best_routes


def apply_crossover(G: nx.MultiDiGraph, sp: ShortestPaths, routes1: list[list[RouteStep]], routes2: list[list[RouteStep]]) -> list[list[RouteStep]]:
    """
    Takes two full sets of routes and returns the crossover between them

    Args:
        G (nx.MultiDiGraph): the graph of the network
        sp (ShortestPaths): shortest paths object corresponding to the graph network
        routes1 (list[list[RouteStep]]): the first set of routes
        routes2 (list[list[RouteStep]]): the second set of routes

    Returns:
        list[list[RouteStep]]: new set of routes
    """

    routes0, change_index = combine(routes1, routes2)
    remove_duplicates(change_index, routes0, sp)

    for edge in get_missing_edges(G, routes0):
        step = RouteStep(edge[0], edge[1], edge[2])
        routes0 = insert_edge(G, step, routes0, sp)
    return routes0