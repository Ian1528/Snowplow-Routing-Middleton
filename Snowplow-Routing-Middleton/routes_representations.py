from shortest_paths import ShortestPaths
class RouteStep:
    """
    Represents a step in a route.

    Attributes:
        node1 (int): The starting node of the step.
        node2 (int): The ending node of the step.
        id (int): The id of the step (identifier for parallel edges)
        deadheaded (bool): Indicates whether the step was a deadhead (non-serviced) move.
        hadOptions (bool): Indicates whether the step had multiple options for routing. Could have gone a different direction
        saltval (int): Indicates the amount of salt the vehicle had before traversing the arc. Useful for checking validity of solution
        next (RouteStep): The next step in the route
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
    
class FullRoute:
    def __init__(self, root: RouteStep=None, length: int=0, cost: int=0, routes_data: list[tuple[RouteStep, RouteStep, int]] = []):
        self.root = root
        self.length = length
        self.cost = cost
        self.routes_data = routes_data
    
    def print_full(self) -> None:
        node: RouteStep = self.root

        while node is not None:
            print(node)
            node = node.next
    
def full_routes(sp: ShortestPaths, routes: list[list[tuple[int, int, int]]]) -> list[tuple[int, int, int]]:
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
                else:
                    path = sp.get_shortest_path(edge, next_edge)
                    full_route.extend(path)
                    full_route.pop()
            else:
                full_route.append(edge)
    return full_route