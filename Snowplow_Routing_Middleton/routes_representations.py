from shortest_paths import ShortestPaths
from params import SALT_CAP
import networkx as nx
class RouteStep:
    """
    Represents a step in a route. 

    Attributes:
        node1 (int): The starting node of the edge.
        node2 (int): The ending node of the edge.
        edge_id (int): The key of the edge in the multigraph.
        deadheaded (bool): Indicates if the step was a deadhead.
        options (bool): Indicates whether there were other edges to possibly traverse at this step.
        saltval (int): The remaining salt value in the truck for the route step.
        next (RouteStep): The next step in the route.
        prev (RouteStep): The previous step in the route.
        route_id (int): The identifier of the route.
        is_route_end (bool): Indicates if this step is the end of the route before returning to the depot.
    Methods:
        __str__(): Returns a string representation of the route step.
        __repr__(): Returns a string representation of the route step.
        get_edge() -> tuple[int, int, int]: Returns the tuple representation of the route step.
    """
    def __init__(self, node1: int=None, node2: int=None, edge_id: int=None, deadheaded: bool=False, options: bool=False, saltval: int = 0, next: "RouteStep"=None, prev: "RouteStep"=None, route_id: int = -1, is_route_end: bool = False):
        self.node1 = node1
        self.node2 = node2
        self.edge_id = edge_id
        self.deadheaded = deadheaded
        self.options = options
        self.saltval = saltval
        self.next = next
        self.prev = prev
        self.route_id = route_id
        self.is_route_end = is_route_end

    def __str__(self):
        if self.next is None:
            next_str = "None"
        else:
            next_str = f"({self.next.node1}, {self.next.node2}, {self.next.edge_id})"
        if self.prev is None:
            prev_str = "None"
        else:
            prev_str = f"({self.prev.node1}, {self.prev.node2}, {self.prev.edge_id})"
        return f"{self.node1} --> {self.node2}. ({self.edge_id}) Salt value: {self.saltval}. Prev: {prev_str}. Next: {next_str}. Route end: {self.is_route_end}"

    def __repr__(self):
        return str(self)
    
    def get_edge(self) -> tuple[int, int, int]:
        """
        Returns the edge information of the route step.

        Returns:
            tuple[int, int, int]: A tuple containing the node1, node2, and edge_id of the route step.
        """
        return (self.node1, self.node2, self.edge_id)
        
def create_full_routes(sp: ShortestPaths, routes: list[list[tuple[int, int, int]]]) -> list[tuple[int, int, int]]:
    """
    Generates a full route by connecting the given routes using the ShortestPaths object.

    Args:
        sp (ShortestPaths): An instance of the ShortestPaths class.
        routes (list[list[tuple[int, int, int]]]): A list of routes, where each route is represented as a list of tuples.

    Returns:
        list[tuple[int, int, int]]: The full route connecting all the given routes.
    """
    full_route = list()
    for i in range(len(routes)):
        for j in range(len(routes[i])):
            edge = routes[i][j]
            next_edge = routes[i][j+1] if j+1 < len(routes[i]) else routes[i+1][0] if i+1 < len(routes) else None
            if next_edge is not None:
                if edge[1] == next_edge[0]:
                    full_route.append(edge)

                # if the next edge is not connected to the current edge, find the shortest path between them
                else:
                    path = sp.get_shortest_path(edge, next_edge)
                    full_route.extend(path)
                    full_route.pop()
            else:
                full_route.append(edge)
    return full_route   

def create_full_routes_with_returns(G: nx.MultiDiGraph, sp: ShortestPaths, routes: list[list[tuple[int, int, int]]], DEPOT: int) -> list[tuple[int, int, int]]:
    """
    Create full routes with returns to the depot when salt runs out.

    This function generates a complete route for snowplowing, ensuring that the 
    snowplow returns to the depot to refill salt when it runs out. It uses the 
    shortest paths to navigate between edges and the depot.

    Args:
        G (nx.MultiDiGraph): The graph representing the road network.
        sp (ShortestPaths): An object that provides shortest path calculations.
        routes (list[list[tuple[int, int, int]]]): A list of routes, where each route 
            is a list of edges represented as tuples (start_node, end_node, key).
        DEPOT (int): The node representing the depot location.

    Returns:
        list[tuple[int, int, int]]: A list of edges representing the full route, including returns to the depot when salt runs out.
    """
    
    full_route = list()
    salt_val = SALT_CAP
    for i in range(len(routes)):
        for j in range(len(routes[i])):
            edge = routes[i][j]
            edge_data = G.get_edge_data(edge[0], edge[1], edge[2])

            # check to see if salt runs out
            if salt_val - edge_data['salt_per'] < 0:
                path = sp.get_shortest_path(edge, (DEPOT, DEPOT, 0))
                full_route.extend(path)
                salt_val = SALT_CAP
                continue

            salt_val -= edge_data['salt_per']
            if j == 0:
                path = sp.get_shortest_path((DEPOT, DEPOT, 0), edge)
                full_route.extend(path)
            elif j+1 < len(routes[i]):
                next_edge = routes[i][j+1]
                if edge[1] == next_edge[0]:
                    full_route.append(edge)
                else:
                    path = sp.get_shortest_path(edge, next_edge)
                    full_route.extend(path)
                    full_route.pop()
            else:
                path = sp.get_shortest_path(edge, (DEPOT, DEPOT, 0))
                full_route.extend(path)
                if i != len(routes)-1:
                    full_route.pop()
    return full_route            
