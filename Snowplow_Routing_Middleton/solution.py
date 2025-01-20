class Solution:
    """
    A class to represent a solution for the snowplow routing problem.
    Attributes:
        routes (list[list[tuple[int, int, int]]]): A list of routes, where each route is a list of tuples representing the edges of the multigraph.
        similarities (dict["Solution", int]): A dictionary mapping other Solution instances to their similarity scores.
        cost (int): The cost associated with this solution.
        totalSimScore (int): The total similarity score of this solution compared to all other solutions.
    Methods:
        __str__(): Returns a string representation of the solution.
        __repr__(): Returns a string representation of the solution.
        add_similarity(S: "Solution", sim: int) -> None: Adds a similarity score between this solution and another solution.
        remove_similarity(S: "Solution") -> None: Removes the similarity score between this solution and another solution.
    """
    def __init__(self, route: list[tuple[int, int, int]], similarities: dict["Solution", int], cost: int, totalSimScore: int):
        """
        Initializes a Solution instance.
        Args:
            routes (list[list[tuple[int, int, int]]]): A list of routes, where each route is a list of tuples representing edges of the multigraph.
            similarities (dict["Solution", int]): A dictionary mapping Solution instances to their similarity scores.
            cost (int): The cost associated with the solution.
            totalSimScore (int): The total similarity score for the solution.
        """
        self.route = route
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
