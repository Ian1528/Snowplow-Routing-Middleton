from routes_representations import RouteStep, FullRoute
from shortest_paths import ShortestPaths
from solution import Solution
from params import DEPOT, K
from costs import routes_cost
import copy
import random
import networkx as nx

def individual_to_linked_list(S: list[list[RouteStep]]) -> tuple[dict[tuple[int, int, int]: RouteStep], RouteStep]:
    """
    Converts a list of routes to a linked list representation. This is useful for local search operators

    Args:
        S (list[list[RouteStep]]): the list of routes

    Returns:
        dict[tuple[int, int, int]: RouteStep], RouteStep: a dictionary of edges to RouteStep objects and the head of the linked list
    """
    routesteps = dict()
    head = None
    # update links
    for i in range(len(S)):
        for j in range(len(S[i])):
            step = S[i][j]
            if j == 0 and i == 0:
                head = step
            step.prev = S[i][j-1] if j > 0 else (S[i-1][-1] if i > 0 else None)
            step.next = S[i][j+1] if j < len(S[i])-1 else (S[i+1][0] if i < len(S)-1 else None)
            
            routesteps[step.get_edge()] = step
    return routesteps, head
def linked_list_to_individual(head: RouteStep) -> list[list[RouteStep]]:
    """
    Converts a linked list representation to a list of routes. Useful for converting back after local search operators

    Args:
        head: Routestep. The routestep that is the head of the linked list (has no previous node)
    Returns:
        list[list[RouteStep]]: the list of routes
    """
    S = []
    step = head
    current_route = []
    while step != None:
        current_route.append(step)
        if step.node2 == DEPOT:
            S.append(current_route)
            current_route = []
        step = step.next
    return S

def find_indice(S: list[list[RouteStep]], edge: tuple[int, int, int]) -> tuple[int, int]:
    """
    Finds the corresponding indice of an edge in a route

    Args:
        S (list[list[RouteStep]]): the list of routes
        edge (tuple[int, int, int]): edge to look for

    Returns:
        tuple[int, int]: tuple containing two indices. The first is the indice of overall route,
        second is indice within that route
    """
    edge = tuple(edge)
    for i in range(len(S)):
        route = S[i]
        for j in range(len(route)):
            if route[j].get_edge() == edge:
                return (i, j)
    
    raise Exception("Edge not found")

def relocate(routesteps: dict[tuple[int, int, int]: RouteStep], edge1: tuple[int, int, int], edge2: tuple[int, int, int], sp: ShortestPaths, threshold: float = 1) -> list[list[RouteStep]]:
    """
    Relocates edge1 after edge2 in the route. Returns the new set of modified routes

    Args:
        routes_old (list[list[RouteStep]]): old set of routes
        edge1 (tuple[int, int, int]): first edge to be moved
        edge2 (tuple[int, int, int]): second edge
        indices1 (tuple[int, int], optional): the indices of the first edge. Defaults to None.
        indices2 (tuple[int, int], optional): indices of the second edge. Defaults to None.
        threshold: threshold of acceptance. If the new cost is less than the old cost * threshold, the move is accepted
    Returns:
        list[list[RouteStep]]: new set of modified routes
    """
    step1 = routesteps[edge1]
    step2 = routesteps[edge2]

    # avoid swapping end edges
    try:
        # precompute the cost of the operation
        cost_old = sp.get_dist(step2.get_edge(), step2.next.get_edge()) + sp.get_dist(step1.get_edge(), step1.next.get_edge()) + sp.get_dist(step1.prev.get_edge(), step1.get_edge()) 
        cost_new = sp.get_dist(step2.get_edge(), step1.get_edge()) + sp.get_dist(step1.get_edge(), step2.next.get_edge()) + sp.get_dist(step1.prev.get_edge(), step1.next.get_edge())

        # if cost is better, proceed with the move. Otherwise, return the old routes
        if cost_new < cost_old * threshold:
            # update all links
            step1.prev.next = step1.next
            step1.next.prev = step1.prev

            step2.next.prev = step1
            step1.next = step2.next

            step1.prev = step2
            step2.next = step1
            return True
        
        return False
    except:
        return False

def relocate_v2(routesteps: dict[tuple[int, int, int]: RouteStep], edge1: tuple[int, int, int], edge2: tuple[int, int, int], sp: ShortestPaths, threshold: float = 1) -> list[list[RouteStep]]:
    """
    Moves edge1 and the edge immediately following after edge2.

    Args:
        routes_old (list[list[RouteStep]]): old set of routes
        edge1 (tuple[int, int, int]): first edge to be moved
        edge2 (tuple[int, int, int]): second edge
        indices1 (tuple[int, int], optional): the indices of the first edge. Defaults to None.
        indices2 (tuple[int, int], optional): indices of the second edge. Defaults to None.

    Returns:
        list[list[RouteStep]]: new set of modified routes
    """
    step1 = routesteps[edge1]
    step2 = routesteps[edge2]

    # no next edge exists
    if step1.next == None:
        return False

    try:
        cost_old = sp.get_dist(step2.get_edge(), step2.next.get_edge()) + sp.get_dist(step1.prev.get_edge(), step1.get_edge()) + sp.get_dist(step1.next.get_edge(), step1.next.next.get_edge())
        cost_new = sp.get_dist(step2.get_edge(), step1.get_edge()) + sp.get_dist(step1.next.get_edge(), step2.next.get_edge()) + sp.get_dist(step1.prev.get_edge(), step1.next.next.get_edge())

        if cost_new < cost_old * threshold:
            step1.prev.next = step1.next.next
            step1.next.next.prev = step1.prev

            step2.next.prev = step1.next
            step1.next.next = step2.next

            step1.prev = step2
            step2.next = step1

            return True
        return False
    except:
        return False

def swap(routesteps: dict[tuple[int, int, int]: RouteStep], edge1: tuple[int, int, int], edge2: tuple[int, int, int], sp: ShortestPaths, threshold: float = 1) -> list[list[RouteStep]]:
    """
    Swaps edge1 with edge2.

    Args:
        routes_old (list[list[RouteStep]]): old set of routes
        edge1 (tuple[int, int, int]): first edge to be moved
        edge2 (tuple[int, int, int]): second edge
        indices1 (tuple[int, int], optional): the indices of the first edge. Defaults to None.
        indices2 (tuple[int, int], optional): indices of the second edge. Defaults to None.

    Returns:
        list[list[RouteStep]]: new set of modified routes
    """
    step1 = routesteps[edge1]
    step2 = routesteps[edge2]
    try:
        cost_old = sp.get_dist(step1.prev.get_edge(), step1.get_edge()) + sp.get_dist(step1.get_edge(), step1.next.get_edge()) + sp.get_dist(step2.prev.get_edge(), step2.get_edge()) + sp.get_dist(step2.get_edge(), step2.next.get_edge())
        cost_new = sp.get_dist(step1.prev.get_edge(), step2.get_edge()) + sp.get_dist(step2.get_edge(), step1.next.get_edge()) + sp.get_dist(step2.prev.get_edge(), step1.get_edge()) + sp.get_dist(step1.get_edge(), step2.next.get_edge())

        if cost_new < cost_old * threshold:
            step1.prev.next = step2
            step1.next.prev = step2

            step2.prev.next = step1
            step2.next.prev = step1

            temp = step1.prev
            step1.prev = step2.prev
            step2.prev = temp

            temp = step1.next
            step1.next = step2.next
            step2.next = temp
            return True
        return False
    except:
        return False
def two_opt(routesteps: dict[tuple[int, int, int]: RouteStep], edge1: tuple[int, int, int], edge2: tuple[int, int, int], sp: ShortestPaths, threshold: float = 1) -> list[list[RouteStep]]:
    """
    Reverse the order of traversal of the edges between the two indicated edges within the route. This is a swap with a reversal
    Example: two opt between b and f:
    a->b->c->d->e->f->g becomes a->f->e->d->c->b->g

    Args:
        routes_old (list[list[RouteStep]]): old set of routes
        edge1 (tuple[int, int, int]): first edge to be moved
        edge2 (tuple[int, int, int]): second edge
        indices1 (tuple[int, int], optional): the indices of the first edge. Defaults to None.
        indices2 (tuple[int, int], optional): indices of the second edge. Defaults to None.

    Returns:
        list[list[RouteStep]]: new set of modified routes
    """
    step1 = routesteps[edge1]
    step2 = routesteps[edge2]
    try:
        old_cost = sp.get_dist(step1.prev.get_edge(), step1.get_edge()) + sp.get_dist(step2.get_edge(), step2.next.get_edge())
        new_cost = sp.get_dist(step1.prev.get_edge(), step2.get_edge()) + sp.get_dist(step1.get_edge(), step2.next.get_edge())
        
        if new_cost < old_cost * threshold:
            # the edges are in the same route. Update all links
            # reversing a linked list from step1 to step2. #TODO: Test this
            curr_step = step1
            temp = None
            final = step2.next
            while curr_step != final:
                temp = curr_step.prev
                curr_step.prev = curr_step.next
                curr_step.next = temp
                curr_step = curr_step.prev

            return True
        
        return False
    
    except:
        return False

def local_improve(S: Solution, G: nx.MultiDiGraph, sp: ShortestPaths, required_edges: set[tuple[int, int, int]], K: int=3) -> Solution:
    """
    Takes a current solution and runs the local improvement algorithm. First, the four local search operators are randomly shuffled.
    Then, for the k-nearest neighbors of each edge, every operator is applied and accepted if it reduces the route cost.

    Args:
        S (Solution): current solution
        G (nx.MultiDiGraph): graph representing the street network
        sp (ShortestPaths): corresponding shortest paths object, needed to compute nearest neighbors
        required_edges (set[tuple[int, int, int]]): set of required edges in the graph network
        K (int, optional): the number of nearest edges to explore. Defaults to 3.

    Returns:
        Solution: the new solution after local improvement
    """
    ALL_EDGES = [step.get_edge() for route in S.routes for step in route if step.get_edge() != (DEPOT,DEPOT,0)]
    operators = [relocate, relocate_v2, swap, two_opt]
    
    S_new = copy.deepcopy(S) # deepcopy so that all the routesteps are copied #TODO: make sure it is deepcopying

    routestep_dict, head = individual_to_linked_list(S_new.routesteps)
    random.shuffle(ALL_EDGES)
    random.shuffle(operators)

    nearest_neighbors = sp.compute_nearest_neighbors()

    for operator in operators:
        for edge in ALL_EDGES:
            for neighboring_edge in nearest_neighbors[edge][0:K]:
                neighboring_edge = tuple(neighboring_edge)

                if neighboring_edge == (DEPOT,DEPOT,0) or neighboring_edge not in required_edges:
                    continue
                
                modified: bool = operator(routestep_dict, edge, neighboring_edge, sp, threshold=1)
                # curr_cost = routes_cost(G, sp, S_curr_routes)
                # if curr_cost < S_best.cost:
                #     S_best = Solution(S_curr_routes, dict(), curr_cost, 0)
    return S_new
