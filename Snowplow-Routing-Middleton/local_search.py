from routes_representations import RouteStep, FullRoute
from shortest_paths import ShortestPaths
from solution import Solution
from params import K, SALT_CAP
from costs import routes_cost, routes_cost_linked_list
import copy
import random
import networkx as nx
import params
class Node:
    def __init__(self, data: tuple[int, int, int], next: "Node" = None, prev: "Node" = None, is_route_end: bool = False, route_belong: str = "OLD"):
        self.data = data
        self.next = next
        self.prev = prev
        self.is_route_end = is_route_end
        self.route_belong = route_belong

    def __str__(self):
        return f"{self.prev.data if self.prev else None} -> [{self.data}] -> {self.next.data if self.next else None} {self.route_belong}"
    def __repr__(self):
        return str(self)

def swap_steps(x: Node, y: Node):
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
    x.is_route_end, y.is_route_end = y.is_route_end, x.is_route_end


def insert(x: Node, y: Node):
    """
    Inserts step x after step y

    Args:
        x (RouteStep): step to be inserted
        y (RouteStep): step to insert after
    """
    oldXprev = x.prev

    x.prev.next = x.next
    x.next.prev = x.prev

    y.next.prev = x
    x.prev = y
    x.next = y.next
    y.next = x

    
    # update route endpoints
    oldXprev.is_route_end = x.is_route_end
    x.is_route_end = y.is_route_end
    y.is_route_end = False



def reverse_list(n1: Node, n2: Node):
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

def individual_to_linked_list(S: list[list[tuple[int, int, int]]], DEPOT: int) -> Node:
    """
    Converts a list of routes to a linked list representation. This is useful for local search operators

    Args:
        S (list[list[RouteStep]]): the list of routes
        DEPOT (int): the depot node

    Returns:
        dict[tuple[int, int, int]: RouteStep], RouteStep: a dictionary of edges to RouteStep objects and the head of the linked list
    """
    head = Node((DEPOT, DEPOT, 0))
    tail = Node((DEPOT, DEPOT, 0))
    edge_node_map = dict()
    prev_node = head
    # update links
    for i in range(len(S)):
        for j in range(len(S[i])):
            new_node = Node(S[i][j])

            if i == 0 and j == 0:
                head.next = new_node
                new_node.prev = head
                prev_node = new_node
            elif i == len(S)-1 and j == len(S[i])-1:
                tail.prev = new_node
                
                new_node.prev = prev_node
                prev_node.next = new_node

                new_node.next = tail
                new_node.is_route_end = True
            else:
                if j == len(S[i])-1:
                    new_node.is_route_end = True
                new_node.prev = prev_node
                prev_node.next = new_node
                prev_node = new_node
            
            edge_node_map[S[i][j]] = new_node

    return edge_node_map, head

def print_linked_list(head: Node):
    node = head.next
    count = 0
    while node != None:
        print(node)
        if node.is_route_end:
            count += 1
        node = node.next
    print("route end counT:", count)

def linked_list_to_individual(head: Node) -> list[list[tuple[int, int, int]]]:
    full_routes = []
    curr_route = []
    node = head.next
    while node != None:
        curr_route.append(node.data)

        if node.is_route_end: # or step.node2 == DEPOT
            full_routes.append(curr_route)
            curr_route = []
        node = node.next

    # if len(curr_route) > 0:
    #     curr_route[-1].is_route_end = True
    #     full_routes.append(curr_route)
    return full_routes


def relocate(G: nx.MultiDiGraph, head: Node, old_cost: float, edge1: Node, edge2: Node, sp: ShortestPaths, DEPOT: int, threshold: float = 1,) -> bool:
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
    edge1prev = edge1.prev

    insert(edge1, edge2)
    new_cost = routes_cost_linked_list(G, sp, head, DEPOT)
    if new_cost < old_cost * threshold:
        return True, new_cost
    else:
        insert(edge1, edge1prev)
        return False, old_cost
    try:
        # precompute the cost of the operation
        cost_old = sp.get_dist(edge2.data, edge2.next.data) + sp.get_dist(edge1.data, edge1.next.data) + sp.get_dist(edge1.prev.data, edge1.data) 
        cost_new = sp.get_dist(edge2.data, edge1.data) + sp.get_dist(edge1.data, edge2.next.data) + sp.get_dist(edge1.prev.data, edge1.next.data)

        # if cost is better, proceed with the move. Otherwise, return the old routes
        if cost_new < cost_old * threshold:
            insert(edge1, edge2)
            return True
        
        # try swapping the other way
        cost_new = sp.get_dist(edge1.data, edge2.data) + sp.get_dist(edge2.data, edge1.next.data) + sp.get_dist(edge2.prev.data, edge2.next.data)
        if cost_new < cost_old * threshold:
            insert(edge2, edge1)
            return True
        return False
    
    except:
        return False

def relocate_v2(G: nx.MultiDiGraph, head: Node, old_cost: float,edge1: Node, edge2: Node, sp: ShortestPaths, DEPOT: int, threshold: float = 1,) -> bool:
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
    edge1_next = edge1.next
    edge1_prev = edge1.prev
    # no next edge exists
    if edge1_next == None or edge1.data == (DEPOT, DEPOT, 0) or edge1_next.data == (DEPOT, DEPOT, 0):
        return False, old_cost

    insert(edge1, edge2)
    insert(edge1_next, edge1)
    new_cost = routes_cost_linked_list(G, sp, head, DEPOT)

    if new_cost < old_cost * threshold:
        return True, new_cost
    else:
        insert(edge1, edge1_prev)
        insert(edge1_next, edge1)
        return False, old_cost
    
    try:
        cost_old = sp.get_dist(edge2.data, edge2.next.data) + sp.get_dist(edge1.prev.data, edge1.data) + sp.get_dist(edge1.next.data, edge1.next.next.data)
        cost_new = sp.get_dist(edge2.data, edge1.data) + sp.get_dist(edge1.next.data, edge2.next.data) + sp.get_dist(edge1.prev.data, edge1.next.next.data)

        if cost_new < cost_old * threshold:
            insert(edge1, edge2)
            insert(edge1_next, edge1)
            return True
        return False
    except:
        return False

def swap(G: nx.MultiDiGraph, head: Node, old_cost: float,edge1: Node, edge2: Node, sp: ShortestPaths, DEPOT: int, threshold: float = 1,) -> bool:
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
    swap_steps(edge1, edge2)
    new_cost = routes_cost_linked_list(G, sp, head, DEPOT)
    if new_cost < old_cost * threshold:
        return True, new_cost
    else:
        swap_steps(edge1, edge2)
        return False, old_cost

def find_route_end_two_steps(edge1: Node, edge2: Node, DEPOT: int) -> tuple[Node, Node] | tuple[None, None]:
    """
    Finds the last step in the route that both steps is a part of, and returns none if step2 comes before step1 in the same route.
    Useful to save time in two-opt intra-route

    Args:
        step (RouteStep): the step

    Returns:
        RouteStep: the last step in the route, or None if step2 comes before step1 in the same route
    """
    curr_step2 = edge2

    while curr_step2.is_route_end == False and curr_step2.next.data != (DEPOT, DEPOT, 0):
        if curr_step2 == edge1:
            return None, None
        curr_step2 = curr_step2.next
    if curr_step2 == edge1:
        return None, None
    curr_step1 = edge1
    while curr_step1.is_route_end == False and curr_step1.next.data != (DEPOT, DEPOT, 0):
        curr_step1 = curr_step1.next

    return curr_step1, curr_step2

def two_opt_intra_route(G: nx.MultiDiGraph, head: Node, old_cost: float, edge1: Node, edge2: Node, edge_node_map: dict, sp: ShortestPaths, DEPOT: int, threshold: float) -> bool:
    """
    Two-opt operator for steps in the same route

    Args:
        step1 (RouteStep): first step
        step2 (RouteStep): second step

    Returns:
        list[list[RouteStep]]: new set of modified routes
    """
    reverse_list(edge1, edge2)
    new_cost = routes_cost_linked_list(G, sp, head, DEPOT)
    if new_cost < old_cost * threshold:
        return head, True, new_cost, edge_node_map
    else:
        reverse_list(edge2, edge1)
        return head, False, old_cost, edge_node_map

def two_opt_inter_route(G: nx.MultiDiGraph, head: Node, old_cost: float, original_edge1: Node, original_edge2: Node, original_edge1end: Node, original_edge2end: Node, original_edge_node_map: dict, sp: ShortestPaths, DEPOT: int, threshold: float) -> bool:
    """
    Two-opt operator for steps in different routes

    Args:
        step1 (RouteStep): first step
        step2 (RouteStep): second step

    Returns:
        list[list[RouteStep]]: new set of modified routes
    """
    # create a copy of the linked list
    new_head = Node((DEPOT, DEPOT, 0), route_belong="New")
    new_edge_node_map = dict()
    prev_node = new_head
    
    node = head.next
    while node != None:
        new_node = Node(node.data, is_route_end=node.is_route_end, route_belong="NEW")
        new_node.prev = prev_node
        prev_node.next = new_node
        prev_node = new_node

        if node.data == original_edge1.data:
            edge1 = new_node
        elif node.data == original_edge2.data:
            edge2 = new_node
        if node.data == original_edge1end.data:
            edge1end = new_node
        if node.data == original_edge2end.data:
            edge2end = new_node

        new_edge_node_map[node.data] = new_node
        node = node.next

    old_edge1end_next = edge1end.next
    old_edge2end_next = edge2end.next

    edge1prev = edge1.prev
    edge2prev = edge2.prev

    edge1.prev.next, edge2.prev.next = edge2, edge1

        
    # handle edge cases where one step is end of one route, other is start of the next
    if edge2prev == edge1end:
        edge1.prev = edge2end
        edge2end.next = edge1
        edge1end.next = old_edge2end_next
        old_edge2end_next.prev = edge1end
    else:
        edge1.prev = edge2prev
        edge2end.next = old_edge1end_next
        old_edge1end_next.prev = edge2end

    if edge1prev == edge2end:
        edge2.prev = edge1end
        edge1end.next = edge2
        edge2end.next = old_edge1end_next
        old_edge1end_next.prev = edge2end
    else:
        edge2.prev = edge1prev
        edge1end.next = old_edge2end_next
        old_edge2end_next.prev = edge1end
    new_cost = routes_cost_linked_list(G, sp, new_head, DEPOT)

    if new_cost < old_cost * threshold:
        return new_head, True, new_cost, new_edge_node_map
    return head, False, old_cost, original_edge_node_map
    
def two_opt(G: nx.MultiDiGraph, head: Node, old_cost: float, edge1: Node, edge2: Node, edge_node_map: dict, sp: ShortestPaths, DEPOT: int, threshold: float = 1) -> list[list[RouteStep]]:
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

    edge1end, edge2end = find_route_end_two_steps(edge1, edge2, DEPOT)

    # None check
    if edge1end == None:
        return head, False, old_cost, edge_node_map

    # if the endedges are the very tail, we want the previous edge
    if edge1end.next == None:
        edge1end = edge1end.prev
    if edge2end.next == None:
        edge2end = edge2end.prev
        
    if edge1end == edge2end:
        return two_opt_intra_route(G, head, old_cost, edge1, edge2, edge_node_map, sp, DEPOT, threshold)
    else:
        return two_opt_inter_route(G, head, old_cost, edge1, edge2, edge1end, edge2end, edge_node_map, sp, DEPOT, threshold)
# def len_routes(head: Node):
#     node = head.next
#     count = 0
#     while node != None and node.data != (DEPOT, DEPOT, 0):
#         count += 1
#         node = node.next
#     return count
def local_improve(S: Solution, G: nx.MultiDiGraph, sp: ShortestPaths, required_edges: set[tuple[int, int, int]], DEPOT: int, threshold: float = 1) -> Solution:
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
    import time
    ALL_EDGES = [edge for route in S.routes for edge in route if edge != (DEPOT,DEPOT,0)]
    operators = [relocate, relocate_v2, swap, two_opt]
    # operators = [two_opt]
    # S_new = copy.deepcopy(S) # deepcopy so that all the routesteps are copied #TODO: make sure it is deepcopying
    edge_node_map, head = individual_to_linked_list(S.routes, DEPOT)
    best_cost = S.cost
    random.shuffle(ALL_EDGES)
    random.shuffle(operators)
    times = []
    times_two_opt = []
    nearest_neighbors = sp.nearest_neighbors
    for edge in ALL_EDGES:
        for neighboring_edge in nearest_neighbors[edge][1:K+1]:
            for operator in operators:
                neighboring_edge = tuple(neighboring_edge)
                if neighboring_edge == (DEPOT,DEPOT,0) or neighboring_edge not in required_edges or neighboring_edge == edge:
                    continue
                starttime = time.time()
                if operator.__name__ == "two_opt":
                    head, modified, best_cost, edge_node_map = operator(G, head, best_cost, edge_node_map[edge], edge_node_map[neighboring_edge], edge_node_map, sp, DEPOT, threshold=threshold)
                else:
                    modified, best_cost = operator(G, head, best_cost, edge_node_map[edge], edge_node_map[neighboring_edge], sp, DEPOT, threshold=threshold)
                endtime = time.time()
                # if operator.__name__ == "two_opt":
                #     times_two_opt.append(endtime-starttime)
                # else:
                #     times.append(endtime-starttime)
                # curr_cost = routes_cost(G, sp, S_curr_routes)
                # if curr_cost < S_best.cost:
                #     S_best = Solution(S_curr_routes, dict(), curr_cost, 0)
    # print("AVerage time of each operator:", sum(times)/len(times), "total time:", sum(times), "total number:", len(times))
    # print("Average time of two opt:", sum(times_two_opt)/len(times_two_opt), "total time:", sum(times_two_opt), "total number:", len(times_two_opt))
    new_routes = linked_list_to_individual(head)
    # print("Best cost in loop:", best_cost, "computed cost:", routes_cost(G, sp, new_routes))
    return Solution(new_routes, S.similarities, best_cost, 0)


if __name__ == "__main__":
    from main import create_instance
    from shortest_paths import ShortestPaths
    from construction import route_generation
    from solution import Solution
    from params import find_depot
    G, G_DUAL = create_instance(("smalltoy", "genetic"))
    DEPOT = find_depot(G)[0]
    shortest_paths = ShortestPaths(G_DUAL, load_data=False, save_data=False)
    r, rreq = route_generation(G, shortest_paths, DEPOT)
    required_edges = set(edge[:3] for edge in G.edges(data=True, keys=True) if edge[3]['priority'] != 0)
    S_first = Solution(rreq, dict(), routes_cost(G, shortest_paths, rreq, DEPOT), 0)
    for route in S_first.routes:
        for step in route:
            print(step)
        print("*********************")
    print("Initial cost: ", S_first.cost)

    S_new = local_improve(S_first, G, shortest_paths, required_edges, DEPOT, threshold=2)
    print("New routes:")
    for route in S_new.routes:
        for step in route:
            print(step)
        print("_____")

