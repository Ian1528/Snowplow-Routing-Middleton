class RouteStep:
    """
    Represents a step in a route.

    Attributes:
        node1 (str): The starting node of the step.
        node2 (str): The ending node of the step.
        id (int): The id of the step (identifier for parallel edges)
        deadheaded (bool): Indicates whether the step was a deadhead (non-serviced) move.
        hadOptions (bool): Indicates whether the step had multiple options for routing. Could have gone a different direction
        saltval (int): Indicates the amount of salt the vehicle had before traversing the arc. Useful for checking validity of solution
    """
    def __init__(self, node1="", node2="", edge_id=0, deadheaded=False, options=False, saltval = 0):
        self.node1 = node1
        self.node2 = node2
        self.edge_id = edge_id
        self.deadheaded = deadheaded
        self.options = options
        self.saltval = saltval

    def __str__(self):
        return f"{self.node1} --> {self.node2}. ({self.edge_id}) Deadhead: {self.deadheaded}. Had options: {self.options}. Salt value: {self.saltval}"

    def __repr__(self):
        return str(self)
    
    def get_edge(self) -> tuple[int, int, int]:
        """
        Returns the edge information of the route step.

        Returns:
            tuple[int, int, int]: A tuple containing the node1, node2, and edge_id of the route step.
        """
        return (self.node1, self.node2, self.edge_id)
    