from routes_representations import RouteStep

class Solution:
    """
    Represents a solution for the snowplow routing problem.
    
    Attributes:
        routes (list[list[RouteStep]]): The list of routes for the solution.
        similarities (dict["Solution", int]): A dictionary of similarities between this solution and other solutions.
        cost (int): The cost of the solution.
        totalSimScore (int): The total similarity score of the solution.
    """
    
    def __init__(self, routesteps: dict[tuple[int, int, int]: RouteStep], routes: list[list[RouteStep]], similarities: dict["Solution", int], cost: int, totalSimScore: int):
        self.routesteps = routesteps
        self.routes = routes
        self.cost = cost
        self.similarities = similarities
        self.totalSimScore = totalSimScore

    def __str__(self):
        return f"Cost: {self.cost}, Similarity Score: {self.totalSimScore}, Similarities: {self.similarities}"
    
    def __repr__(self):
        return str(self)
    
    def add_similarity(self, S: "Solution", sim) -> None:
        """
        Adds a similarity score between this solution and another solution.
        
        Args:
            S (Solution): The other solution to compare with.
            sim (int): The similarity score between the solutions.
        """
        self.similarities[S] = sim
        self.totalSimScore += sim
        
    def remove_similarity(self, S: "Solution") -> None:
        """
        Removes the similarity score between this solution and another solution.
        
        Args:
            S (Solution): The other solution to remove the similarity score for.
        """
        self.totalSimScore -= self.similarities[S]
        del self.similarities[S]
