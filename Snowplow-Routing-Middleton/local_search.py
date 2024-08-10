from routes_representations import RouteStep, FullRoute
from shortest_paths import ShortestPaths
from solution import Solution
from params import DEPOT, K, SALT_CAP
from costs import routes_cost
import copy
import random
import networkx as nx

def swap_steps(x: RouteStep, y: RouteStep):
    """
    Swaps two routesteps

    Args:
        x (RouteStep): first routestep
        y (RouteStep): second routestep
    """
    xNext = x.next
    xPrev = x.prev
    yNext = y.next
    yPrev = y.prev

    if xPrev:
        xPrev.next = y
    if xNext:
        xNext.prev = y
    if yPrev:
        yPrev.next = x
    if yNext:
        yNext.prev = x

    # case 1: x is right before y
    if xNext == y:
        x.next = yNext
        x.prev = y
        y.next = x
        y.prev = xPrev

    
    # case 2: y is right before x
    elif yNext == x:
        x.next = y
        x.prev = yPrev
        y.next = xNext
        y.prev = x

    # case 3: non-adjacent
    else:
        x.prev = yPrev
        x.next = yNext
        y.prev = xPrev
        y.next = xNext
    
    # update the route_id of the steps
    x.route_id, y.route_id = y.route_id, x.route_id
    x.is_route_end, y.is_route_end = y.is_route_end, x.is_route_end

    if x.is_route_end:
        if not y.is_route_end:
            x.is_route_end = False
        y.is_route_end = True

    if y.is_route_end:
        if not x.is_route_end:
            y.is_route_end = False
        x.is_route_end = True
    # if x.node2 == DEPOT:
    #     x.is_route_end = True
    # if y.node2 == DEPOT:
    #     y.is_route_end = True

    # if x.node1 == DEPOT and x.prev != None and x.prev.get_edge() != (DEPOT, DEPOT, 0):
    #     x.prev.is_route_end = True
    #     if y.node1 != DEPOT:
    #         y.prev.is_route_end = False
    
    # if y.node1 == DEPOT and y.prev != None and y.prev.get_edge() != (DEPOT, DEPOT, 0):
    #     y.prev.is_route_end = True
    #     if x.node1 != DEPOT:
    #         x.prev.is_route_end = False


def insert(x: RouteStep, y: RouteStep):
    """
    Inserts step x after step y

    Args:
        x (RouteStep): step to be inserted
        y (RouteStep): step to insert after
    """
    oldXprev = x.prev

    x.prev.next = x.next
    x.next.prev = x.prev

    # update route endpoints
    if x.is_route_end:
        x.prev.is_route_end = True

    if y.is_route_end:
        x.is_route_end = True
        y.is_route_end = False
    else:
        x.is_route_end = False

    y.next.prev = x
    x.prev = y
    x.next = y.next
    y.next = x

    # update the route id of the steps
    x.route_id = y.route_id

    if x.is_route_end:
        oldXprev.is_route_end = True
        if y.is_route_end:
            y.is_route_end = False
        else:
            x.is_route_end = False
    # if x.node2 == DEPOT:
    #     x.is_route_end = True
    # if y.node2 == DEPOT:
    #     y.is_route_end = True

    # if x.node1 == DEPOT:
    #     y.is_route_end = True
    #     if oldXprev.next != None and oldXprev.next.get_edge() != (DEPOT, DEPOT, 0):
    #         oldXprev.is_route_end = False


def reverse_list(n1: RouteStep, n2: RouteStep):
    """
    Reverses the linked list between node1 and node2, inclusive. Node 1 comes before node 2.
    Assumes that node1 and node2 aren't the head or tail of the linked list. Never touching the dummy heads and tails
    Args:
        node1 (RouteStep): The first routestep.
        node2 (RouteStep): The second routestep.

    Returns:
        None
    """
    if n1 == n2:
        return
    
    if n2.is_route_end:
        n2.is_route_end = False
        n1.is_route_end = True
    
    original_prev = n1.prev
    original_next = n2.next
    current = n1
    final = n2.next

    while current != final:
        old_next = current.next
        old_prev = current.prev

        # don't update extra pointers for n1 and n2
        if current == n1:
            current.prev =  old_next
        elif current == n2:
            current.next = old_prev
        else:
            current.next = old_prev
            current.prev = old_next
        current = old_next
    
    n2.prev = original_prev
    original_prev.next = n2

    n1.next = original_next 
    original_next.prev = n1

    if n2.is_route_end:
        n2.is_route_end = False
        n1.is_route_end = True

    # if n1.node2 == DEPOT:
    #     n1.is_route_end = True
    # if n2.node2 == DEPOT:
    #     n2.is_route_end = True
debugging_list = []
def individual_to_linked_list(S: list[list[RouteStep]]) -> tuple[dict[tuple[int, int, int]: RouteStep], RouteStep]:
    """
    Converts a list of routes to a linked list representation. This is useful for local search operators

    Args:
        S (list[list[RouteStep]]): the list of routes

    Returns:
        dict[tuple[int, int, int]: RouteStep], RouteStep: a dictionary of edges to RouteStep objects and the head of the linked list
    """
    global debugging_list
    debugging_list = []
    routesteps = dict()
    head = RouteStep(DEPOT, DEPOT, 0, False, False, SALT_CAP, None, None, -1)
    tail = RouteStep(DEPOT, DEPOT, 0, False, False, SALT_CAP, None, None, -1)
    debugging_list.append(head)
    debugging_list.append(tail)
    # update links
    for i in range(len(S)):
        for j in range(len(S[i])):
            step = S[i][j]

            if i == 0 and j == 0:
                head.next = step
            if i == len(S)-1 and j == len(S[i])-1:
                tail.prev = step

            if j == len(S[i])-1:
                step.is_route_end = True
            
            step.prev = S[i][j-1] if j > 0 else (S[i-1][-1] if i > 0 else head)
            step.next = S[i][j+1] if j < len(S[i])-1 else (S[i+1][0] if i < len(S)-1 else tail)
            routesteps[step.get_edge()] = step
            debugging_list.append(step)
    return routesteps, head

def linked_list_to_individual(head: RouteStep) -> list[list[RouteStep]]:
    full_routes = []
    curr_route = []
    step = head.next
    while step != None:
        if step.prev.get_edge() == (DEPOT, DEPOT, 0):
            step.prev = None
        if step.next.get_edge() == (DEPOT, DEPOT, 0):
            step.next = None

        curr_route.append(step)

        if step.is_route_end: # or step.node2 == DEPOT
            full_routes.append(curr_route)
            curr_route = []
        step = step.next

    if len(curr_route) > 0:
        full_routes.append(curr_route)
    return full_routes

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

    try:
        # precompute the cost of the operation
        cost_old = sp.get_dist(step2.get_edge(), step2.next.get_edge()) + sp.get_dist(step1.get_edge(), step1.next.get_edge()) + sp.get_dist(step1.prev.get_edge(), step1.get_edge()) 
        cost_new = sp.get_dist(step2.get_edge(), step1.get_edge()) + sp.get_dist(step1.get_edge(), step2.next.get_edge()) + sp.get_dist(step1.prev.get_edge(), step1.next.get_edge())

        # if cost is better, proceed with the move. Otherwise, return the old routes
        if cost_new < cost_old * threshold:
            insert(step1, step2)
            return True
        
        # try swapping the other way
        cost_new = sp.get_dist(step1.get_edge(), step2.get_edge()) + sp.get_dist(step2.get_edge(), step1.next.get_edge()) + sp.get_dist(step2.prev.get_edge(), step2.next.get_edge())
        if cost_new < cost_old * threshold:
            insert(step2, step1)
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

    step1Next = step1.next
    # no next edge exists
    if step1Next == None or step1Next.get_edge() == (DEPOT, DEPOT, 0):
        return False
    try:
        cost_old = sp.get_dist(step2.get_edge(), step2.next.get_edge()) + sp.get_dist(step1.prev.get_edge(), step1.get_edge()) + sp.get_dist(step1.next.get_edge(), step1.next.next.get_edge())
        cost_new = sp.get_dist(step2.get_edge(), step1.get_edge()) + sp.get_dist(step1.next.get_edge(), step2.next.get_edge()) + sp.get_dist(step1.prev.get_edge(), step1.next.next.get_edge())

        if cost_new < cost_old * threshold:
            insert(step1, step2)
            insert(step1Next, step1)
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
            swap_steps(step1, step2)
            return True
        return False
    except:
        return False

def find_route_end(step: RouteStep) -> RouteStep:
    """
    Finds the last step in the route that the step is a part of

    Args:
        step (RouteStep): the step

    Returns:
        RouteStep: the last step in the route
    """
    curr_step = step
    while curr_step.is_route_end == False:
        curr_step = curr_step.next
    return curr_step

def print_linked_list(head: RouteStep):
    """
    Prints the linked list

    Args:
        head (RouteStep): the head of the linked list
    """
    curr = head
    while curr != None and curr.is_route_end == False:
        print(curr)
        curr = curr.next
    if curr != None:
        print(curr)

def find_route_end_two_steps(step1: RouteStep, step2: RouteStep) -> tuple[RouteStep, RouteStep] | tuple[None, None]:
    """
    Finds the last step in the route that the both steps is a part of, and returns none if step2 comes before step1 in the same route.
    Useful to save time in two-opt intra-route

    Args:
        step (RouteStep): the step

    Returns:
        RouteStep: the last step in the route, or None if step2 comes before step1 in the same route
    """
    curr_step2 = step2

    while curr_step2.is_route_end == False:
        if curr_step2 == step1:
            return None, None
        curr_step2 = curr_step2.next
    if curr_step2 == step1:
        return None, None
    curr_step1 = step1
    # somehow, the tail is getting moved when it should never be touched.
    while curr_step1.is_route_end == False:
        curr_step1 = curr_step1.next

    return curr_step1, curr_step2

def two_opt_intra_route(step1, step2, sp: ShortestPaths, threshold: float) -> bool:
    """
    Two-opt operator for steps in the same route

    Args:
        step1 (RouteStep): first step
        step2 (RouteStep): second step

    Returns:
        list[list[RouteStep]]: new set of modified routes
    """

    old_cost = sp.get_dist(step1.prev.get_edge(), step1.get_edge()) + sp.get_dist(step2.get_edge(), step2.next.get_edge()) #+ sp.get_dist(step2.prev.get_edge(), step2.get_edge()) + sp.get_dist(step1.get_edge(), step1.next.get_edge())
    new_cost = sp.get_dist(step1.prev.get_edge(), step2.get_edge()) + sp.get_dist(step1.get_edge(), step2.next.get_edge()) #+ sp.get_dist(step2.prev.get_edge(), step1.get_edge()) + sp.get_dist(step2.get_edge(), step1.next.get_edge())

    if new_cost < old_cost * threshold:
        # reverse the sequence of steps between step1 and step2
        reverse_list(step1, step2)
        return True
    return False

def two_opt_inter_route(step1, step2, step1end, step2end, sp: ShortestPaths, threshold: float) -> bool:
    """
    Two-opt operator for steps in different routes

    Args:
        step1 (RouteStep): first step
        step2 (RouteStep): second step

    Returns:
        list[list[RouteStep]]: new set of modified routes
    """
    old_cost = sp.get_dist(step1.prev.get_edge(), step1.get_edge()) + sp.get_dist(step2.prev.get_edge(), step2.get_edge())
    new_cost = sp.get_dist(step1.prev.get_edge(), step2.get_edge()) + sp.get_dist(step2.prev.get_edge(), step1.get_edge())
    
    if new_cost < old_cost * threshold:
        old_step1end_next = step1end.next
        old_step2end_next = step2end.next

        step1prev = step1.prev
        step2prev = step2.prev

        step1.prev.next, step2.prev.next = step2, step1

            
        # handle edge cases where one step is end of one route, other is start of the next
        if step2prev == step1end:
            step1.prev = step2end
            step2end.next = step1
            step1end.next = old_step2end_next
            old_step2end_next.prev = step1end
        else:
            step1.prev = step2prev
            step2end.next = old_step1end_next
            old_step1end_next.prev = step2end
        
        if step1prev == step2end:
            step2.prev = step1end
            step1end.next = step2
            step2end.next = old_step1end_next
            old_step1end_next.prev = step2end
        else:
            step2.prev = step1prev
            step1end.next = old_step2end_next
            old_step2end_next.prev = step1end
        return True
    return False
def two_opt(routesteps: dict[tuple[int, int, int]: RouteStep], edge1: tuple[int, int, int], edge2: tuple[int, int, int], sp: ShortestPaths, threshold: float = 1) -> list[list[RouteStep]]:
    """
    Reverse the order of traversal of the edges between the two indicated edges within the route. This is a swap with a reversal
    Example: two opt between b and f:
    a->b->c->d->e->f->g becomes a->f->e->d->c->b->g

    If the edges belong to different routes, the two next edges are swapped (including their succesccors)
    Example: two opt between b and q:
    a->b->c->d->e->f->g and p->q->r->s->t->u->v becomes a->b->r->s->t->u->v and p->q->c->d->e->f->g

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

    step1end, step2end = find_route_end_two_steps(step1, step2)

    # None check
    if step1end == None:
        return False

    # print("Route from step1: ")
    # print_linked_list(step1)
    # print("Route from step2: ")
    # print_linked_list(step2)
    # print()
    # print()
    # if the endsteps are the very tail, we want the previous step
    if step1end.next == None:
        step1end = step1end.prev
    if step2end.next == None:
        step2end = step2end.prev
        
    if step1end == step2end:
        return two_opt_intra_route(step1, step2, sp, threshold)
    else:
        return two_opt_inter_route(step1, step2, step1end, step2end, sp, threshold)
    
def local_improve(S: Solution, G: nx.MultiDiGraph, sp: ShortestPaths, required_edges: set[tuple[int, int, int]], K: int=3, threshold: float = 1) -> Solution:
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

    routestep_dict, all_routes_original = individual_to_linked_list(S_new.routes)

    random.shuffle(ALL_EDGES)
    random.shuffle(operators)

    nearest_neighbors = sp.compute_nearest_neighbors()
    for operator in operators:
        for edge in ALL_EDGES:
            for neighboring_edge in nearest_neighbors[edge][1:K+1]:
                neighboring_edge = tuple(neighboring_edge)
                if neighboring_edge == (DEPOT,DEPOT,0) or neighboring_edge not in required_edges or neighboring_edge == edge:
                    continue
                modified: bool = two_opt(routestep_dict, edge, neighboring_edge, sp, threshold=threshold) #TODO: change back to operator
                # if modified:
                #     print("Modified")
                # curr_cost = routes_cost(G, sp, S_curr_routes)
                # if curr_cost < S_best.cost:
                #     S_best = Solution(S_curr_routes, dict(), curr_cost, 0)
    S_new.routes = linked_list_to_individual(all_routes_original)
    return S_new


if __name__ == "__main__":
    from main import create_instance
    from shortest_paths import ShortestPaths
    from construction import route_generation
    from solution import Solution
    G, G_DUAL = create_instance(("smalltoy", "genetic"))
    shortest_paths = ShortestPaths(G_DUAL, load_data=False, save_data=False)
    r, rreq = route_generation(G, shortest_paths)
    required_edges = set(edge[:3] for edge in G.edges(data=True, keys=True) if edge[3]['priority'] != 0)
    S_first = Solution(rreq, dict(), routes_cost(G, shortest_paths, rreq), 0)
    for route in S_first.routes:
        for step in route:
            print(step)
        print("*********************")
    print("Initial cost: ", S_first.cost)

    S_new = local_improve(S_first, G, shortest_paths, required_edges, K=3, threshold=2)
    print("New routes:")
    for route in S_new.routes:
        for step in route:
            print(step)
        print("_____")

