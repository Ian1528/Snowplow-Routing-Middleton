"""
This module provides functions for local search operators and the local improvement algorithm.

Functions:
    local_improve(S: Solution, G: nx.MultiDiGraph, sp: ShortestPaths, required_edges: set[tuple[int, int, int]], DEPOT: int, threshold: float = 1) -> Solution:
    - Takes a current solution and runs the local improvement algorithm.
"""
from .shortest_paths import ShortestPaths
from .solution import Solution
from .params import K
from .costs import routes_cost, routes_cost_linked_list
import random
import networkx as nx
class Node:
    """
    A class used to represent a Node in a doubly linked list.

    Attributes
    ----------
    data : tuple[int, int, int]
        A tuple containing three integers representing the edge of the graph.
    next : Node, optional
        A reference to the next node in the list (default is None).
    prev : Node, optional
        A reference to the previous node in the list (default is None).
    is_route_end : bool, optional
        A boolean indicating if the node is the end of a route (default is False).
    
    Methods
    -------
    __str__():
        Returns a string representation of the node and its connections.
    __repr__():
        Returns a string representation of the node and its connections.
    """
    
    def __init__(self, data: tuple[int, int, int], next: "Node" = None, prev: "Node" = None, is_route_end: bool = False):
        self.data = data
        self.next = next
        self.prev = prev
        self.is_route_end = is_route_end

    def __str__(self):
        return f"{self.prev.data if self.prev else None} -> [{self.data}] -> {self.next.data if self.next else None}"
    def __repr__(self):
        return str(self)

def swap_steps(x: Node, y: Node):
    """
    Swaps two nodes in a doubly linked list.

    Args:
        x (Node): first node
        y (Node): second node
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
    Inserts node x after node y

    Args:
        x (Node): Node to be inserted
        y (Node): Node to insert after
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
    Assumes that node1 and node2 aren't the head or tail of the linked list, since we never touch the dummy heads and tails
    
    Args:
        node1 (Node): The first node.
        node2 (Node): The second node.

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

def individual_to_linked_list(S: list[list[tuple[int, int, int]]], DEPOT: int) -> tuple[dict[tuple[int, int, int], Node], Node]:
    """
    Converts a list of routes to a linked list representation.

    Args:
        S (list[list[RouteStep]]): the list of routes
        DEPOT (int): the depot node

    Returns:
        tuple: A tuple containing:
        - edge_node_map (dict[tuple[int, int, int], Node]): A dictionary mapping each route step to its corresponding Node.
        - head (Node): The head node of the linked list.
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

def print_linked_list(head: Node) -> None:
    """
    Prints the linked list starting from the head

    Args:
        head (Node): head of the linked list
    """
    node = head.next
    count = 0
    while node != None:
        print(node)
        if node.is_route_end:
            count += 1
        node = node.next

def linked_list_to_individual(head: Node) -> list[list[tuple[int, int, int]]]:
    """
    Converts a linked list of nodes into a list of routes.
    Each node in the linked list contains data and a flag indicating the end of a route.
    This function traverses the linked list, collects the data from each node, and groups
    them into routes based on the end-of-route flag.
    Args:
        head (Node): The head node of the linked list.
    Returns:
        list[list[tuple[int, int, int]]]: A list of routes, where each route is a list of tuples.
    """

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


def relocate(G: nx.MultiDiGraph, head: Node, old_cost: float, edge1: Node, edge2: Node, sp: ShortestPaths, DEPOT: int, threshold: float = 1,) -> tuple[bool, float]:
    """
    Relocates edge1 after edge2 in the route. Returns whether the move was accepted and the cost of the total route.

    Args:
        G (nx.MultiDiGraph): The graph representing the routes.
        head (Node): The head node of the linked list representing the route.
        old_cost (float): The current cost of the route.
        edge1 (Node): The edge to be relocated.
        edge2 (Node): The edge after which edge1 will be relocated.
        sp (ShortestPaths): An object to compute shortest paths.
        DEPOT (int): The depot node identifier.
        threshold (float, optional): The threshold for accepting the new route cost. Defaults to 1.
    Returns:
        tuple[bool, float]: A tuple containing a boolean indicating whether the move was accepted and the new cost of the route.
    """
    edge1prev = edge1.prev

    insert(edge1, edge2)
    new_cost = routes_cost_linked_list(G, sp, head, DEPOT)
    if new_cost < old_cost * threshold:
        return True, new_cost
    else:
        insert(edge1, edge1prev)
        return False, old_cost

def relocate_v2(G: nx.MultiDiGraph, head: Node, old_cost: float, edge1: Node, edge2: Node, sp: ShortestPaths, DEPOT: int, threshold: float = 1,) -> tuple[bool, float]:
    """
    Inserts edge1 and the next edge after edge2 in the route. Returns whether the move was accepted and the cost of the total route.

    Args:
        G (nx.MultiDiGraph): The graph representing the routes.
        head (Node): The head node of the linked list representing the route.
        old_cost (float): The current cost of the route.
        edge1 (Node): The edge to be relocated.
        edge2 (Node): The edge after which edge1 will be relocated.
        sp (ShortestPaths): An object to compute shortest paths.
        DEPOT (int): The depot node identifier.
        threshold (float, optional): The threshold for accepting the new route cost. Defaults to 1.
    Returns:
        tuple[bool, float]: A tuple containing a boolean indicating whether the move was accepted and the new cost of the route.
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
    
def swap(G: nx.MultiDiGraph, head: Node, old_cost: float,edge1: Node, edge2: Node, sp: ShortestPaths, DEPOT: int, threshold: float = 1,) -> tuple[bool, float]:
    """
    Swaps edge1 with edge2. Returns whether the move was accepted and the cost of the total route.

    Args:
        G (nx.MultiDiGraph): The graph representing the routes.
        head (Node): The head node of the linked list representing the route.
        old_cost (float): The current cost of the route.
        edge1 (Node): The first edge to be swapped.
        edge2 (Node): The second edge to be swapped.
        sp (ShortestPaths): An object to compute shortest paths.
        DEPOT (int): The depot node identifier.
        threshold (float, optional): The threshold for accepting the new route cost. Defaults to 1.

    Returns:
        tuple[bool, float]: A tuple where the first element is a boolean indicating whether the swap was accepted,
        and the second element is the cost of the total route after the swap.

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
    Finds the last step in the route that two edges are a part of, and returns none if step2 comes before step1 in the same route.
    Useful to save time in two-opt intra-route for early termination.

    Args:
        edge1 (Node): first edge
        edge2 (Node): second edge
        DEPOT (int): depot node
    Returns:
        tuple[Node, Node]: The last step in the route that edge1 and edge2 are a part of. None if edge2 comes before edge1 in the same route.
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

def two_opt_intra_route(G: nx.MultiDiGraph, head: Node, old_cost: float, edge1: Node, edge2: Node, edge_node_map: dict[tuple[int,int,int], Node], sp: ShortestPaths, DEPOT: int, threshold: float) -> tuple[Node, bool, float, dict]:
    """
    Two-opt operator for steps in the same route. Called by the two_opt function.
     
    Reverses the order of traversal of the edges between the two indicated edges within the route.

    Args:
        G (nx.MultiDiGraph): The graph representing the routes.
        head (Node): The head node of the linked list representing the route.
        old_cost (float): The current cost of the route.
        edge1 (Node): The first edge to be swapped.
        edge2 (Node): The second edge to be swapped.
        edge_node_map (dict[tuple[int, int, int], Node]): A dictionary mapping each edge to its corresponding Node.
        sp (ShortestPaths): An object to compute shortest paths.
        DEPOT (int): The depot node identifier.
        threshold (float, optional): The threshold for accepting the new route cost. Defaults to 1.
    Returns:
        tuple[Node, bool, float, dict]: A tuple containing:
        - head (Node): The head node of the linked list representing the route.
        - bool: A boolean indicating whether the move was accepted.
        - float: The new cost of the total route.
        - dict: A dictionary mapping each edge to its corresponding Node.
    """
    reverse_list(edge1, edge2)
    new_cost = routes_cost_linked_list(G, sp, head, DEPOT)
    if new_cost < old_cost * threshold:
        return head, True, new_cost, edge_node_map
    else:
        reverse_list(edge2, edge1)
        return head, False, old_cost, edge_node_map

def two_opt_inter_route(G: nx.MultiDiGraph, head: Node, old_cost: float, original_edge1: Node, original_edge2: Node, original_edge1end: Node, original_edge2end: Node, original_edge_node_map: dict, sp: ShortestPaths, DEPOT: int, threshold: float) -> tuple[Node, bool, float, dict]:
    """
    Two-opt operator for steps in different routes. Called by the two_opt function.
    
    Swaps the edges (and all of their successors) between the two routes.

    Args:
        G (nx.MultiDiGraph): The graph representing the routes.
        head (Node): The head node of the linked list representing the route.
        old_cost (float): The current cost of the route.
        original_edge1 (Node): The first edge to be swapped.
        original_edge2 (Node): The second edge to be swapped.
        original_edge1end (Node): The end of the route that the first edge is in.
        original_edge2end (Node): The end of the route that the second edge is in.
        original_edge_node_map (dict[tuple[int, int, int], Node]): A dictionary mapping each edge to its corresponding Node.
        sp (ShortestPaths): An object to compute shortest paths.
        DEPOT (int): The depot node identifier.
        threshold (float, optional): The threshold for accepting the new route cost. Defaults to 1.
    Returns:
        tuple[Node, bool, float, dict]: A tuple containing:
        - head (Node): The head node of the linked list representing the route.
        - bool: A boolean indicating whether the move was accepted.
        - float: The new cost of the total route.
        - dict: A dictionary mapping each edge to its corresponding Node.
    """

    # create a copy of the linked list
    new_head = Node((DEPOT, DEPOT, 0))
    new_edge_node_map = dict()
    prev_node = new_head
    
    node = head.next
    while node != None:
        new_node = Node(node.data, is_route_end=node.is_route_end)
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

    # swap the edges between the routes, including their successors
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
    
def two_opt(G: nx.MultiDiGraph, head: Node, old_cost: float, edge1: Node, edge2: Node, edge_node_map: dict, sp: ShortestPaths, DEPOT: int, threshold: float = 1) -> tuple[Node, bool, float, dict]:
    """
    Reverse the order of traversal of the edges between the two indicated edges within the route. This is a swap with a reversal
    Example: two opt between b and f:
    a->b->c->d->e->f->g becomes a->f->e->d->c->b->g

    If the edges belong to different routes, the two next edges are swapped (including their succesccors)
    Example: two opt between b and q:
    a->b->c->d->e->f->g and p->q->r->s->t->u->v becomes a->b->r->s->t->u->v and p->q->c->d->e->f->g

    Args:
        G (nx.MultiDiGraph): The graph representing the routes.
        head (Node): The head node of the linked list representing the route.
        old_cost (float): The current cost of the route.
        edge1 (Node): The first edge to be swapped.
        edge2 (Node): The second edge to be swapped.
        edge_node_map (dict[tuple[int, int, int], Node]): A dictionary mapping each edge to its corresponding Node.
        sp (ShortestPaths): An object to compute shortest paths.
        DEPOT (int): The depot node identifier.
        threshold (float, optional): The threshold for accepting the new route cost. Defaults to 1.

    Returns:
        Returns:
        tuple[Node, bool, float, dict]: A tuple containing:
        - head (Node): The head node of the linked list representing the route.
        - bool: A boolean indicating whether the move was accepted.
        - float: The new cost of the total route.
        - dict: A dictionary mapping each edge to its corresponding Node.
    """

    edge1end, edge2end = find_route_end_two_steps(edge1, edge2, DEPOT)

    # None check
    if edge1end == None:
        return head, False, old_cost, edge_node_map

    # if the endedges are the tail of the last route, we want the previous edge
    if edge1end.next == None:
        edge1end = edge1end.prev
    if edge2end.next == None:
        edge2end = edge2end.prev
        
    if edge1end == edge2end:
        return two_opt_intra_route(G, head, old_cost, edge1, edge2, edge_node_map, sp, DEPOT, threshold)
    else:
        return two_opt_inter_route(G, head, old_cost, edge1, edge2, edge1end, edge2end, edge_node_map, sp, DEPOT, threshold)

def local_improve(S: Solution, G: nx.MultiDiGraph, sp: ShortestPaths, required_edges: set[tuple[int, int, int]], DEPOT: int, threshold: float = 1) -> Solution:
    """
    Takes a current solution and runs the local improvement algorithm. 
    
    First, the four local search operators are randomly shuffled. Then, for the k-nearest neighbors of each edge, 
    every operator is applied and accepted if it reduces the route cost.

    Args:
        S (Solution): current solution
        G (nx.MultiDiGraph): graph representing the street network
        sp (ShortestPaths): corresponding shortest paths object, needed to compute nearest neighbors
        required_edges (set[tuple[int, int, int]]): set of required edges in the graph network
        DEPOT (int): depot node identifier
        threshold (float, optional): threshold for accepting a new route cost. Defaults to 1.
    Returns:
        Solution: the new solution after local improvement
    """
    ALL_EDGES = [edge for route in S.routes for edge in route if edge != (DEPOT,DEPOT,0)]
    operators = [relocate, relocate_v2, swap, two_opt]
    edge_node_map, head = individual_to_linked_list(S.routes, DEPOT)
    best_cost = S.cost
    random.shuffle(ALL_EDGES)
    random.shuffle(operators)
    nearest_neighbors = sp.nearest_neighbors
    modified_count = 0
    for edge in ALL_EDGES:
        for neighboring_edge in nearest_neighbors[edge][1:K+1]:
            neighboring_edge = tuple(neighboring_edge)
            for operator in operators:
                if neighboring_edge == (DEPOT,DEPOT,0) or neighboring_edge not in required_edges or neighboring_edge == edge:
                    continue
                if operator.__name__ == "two_opt":
                    head, modified, best_cost, edge_node_map = operator(G, head, best_cost, edge_node_map[edge], edge_node_map[neighboring_edge], edge_node_map, sp, DEPOT, threshold=threshold)
                else:
                    modified, best_cost = operator(G, head, best_cost, edge_node_map[edge], edge_node_map[neighboring_edge], sp, DEPOT, threshold=threshold)
                if modified:
                    modified_count += 1
    new_routes = linked_list_to_individual(head)
    return Solution(new_routes, S.similarities, best_cost, 0)
