import networkx as nx
import numpy as np
import pickle
from params import DEPOT
import os

class ShortestPaths:
    """
    Class to compute and store shortest paths in a graph.
    """

    def __init__(self, G_DUAL, load_data=True, save_data=False):
        self.G_DUAL: nx.MultiDiGraph = G_DUAL
        self.edge_index_dict = {edge:index for index, edge in enumerate(self.G_DUAL.nodes)}
        self.index_edge_dict = {index:edge for index, edge in enumerate(self.G_DUAL.nodes)}
        self.predecessors = None
        self.dists_array = None
        self.nearest_neighbors = None

        if load_data:
            self.load_pred_and_dist()        
            self.load_dists_array()
            if not self.params_match_graph():
                print("Params don't match graph. Advised to recompute paths")

        if self.predecessors is None:
            self.compute_pred_and_dist()
        
        if self.dists_array is None:
            self.compute_dists_array()
        
        if save_data:
            self.save_pred_and_dist()
            self.save_dists_array()

        self.nearest_neighbors = self.compute_nearest_neighbors()
    
    def is_none(self) -> bool:
        """
        Checks if the parameters are None.

        Returns:
            bool: True if the parameters are None, False otherwise.
        """

        return self.predecessors is None or self.dists_array is None
    
    def params_match_graph(self) -> bool:
        """
        Checks if the loaded parameters match the graph of the current instance.

        Returns:
            bool: True if the parameters match the graph, False otherwise.
        """
        if self.is_none():
            return False
        check1 = self.dists_array.shape[0] == len(self.G_DUAL.nodes)
        check2 = len(self.predecessors) == len(self.G_DUAL.nodes)
        check3 = (DEPOT, DEPOT, 0) in self.G_DUAL.nodes

        return check1 and check2 and check3
    def load_dists_array(self, filename="C:\\Users\\Sneez\\Desktop\\Snowplowing\\Snowplow-Routing-Middleton\\Snowplow-Routing-Middleton\\graph_data\\shortest_distances_array.npy"):
        """
        Loads the distance matrix from a NumPy file.

        Args:
            filename (str): The path to the NumPy file.

        Raises:
            Exception: If there is no data to load.
        """
        # filepath = os.path.join(os.path.dirname(os.path.realpath('__file__')), filename)

        try:
            self.dists_array = np.load(filename)
        except:
            self.dists_array = None
            print("No data to fetch")
    
    def save_dists_array(self, filename="C:\\Users\\Sneez\\Desktop\\Snowplowing\\Snowplow-Routing-Middleton\\Snowplow-Routing-Middleton\\graph_data\\shortest_distances_array.npy"):
        """
        Saves the distance matrix to a NumPy file.

        Args:
            filename (str): The path to save the NumPy file.

        Raises:
            Exception: If there is no data to save. First compute the distance matrix before saving.
        """
        # filepath = os.path.join(os.path.dirname(os.path.realpath('__file__')), filename)

        if self.dists_array is None:
            raise Exception("No data to save. First compute the distance matrix before saving")
        np.save(filename, self.dists_array)

    def load_pred_and_dist(self, filename="C:\\Users\\Sneez\\Desktop\\Snowplowing\\Snowplow-Routing-Middleton\\Snowplow-Routing-Middleton\\graph_data\\preds_and_dists.pickle"):
        """
        Loads the predecessors and distances from a pickle file.

        Args:
            filename (str): The path to the pickle file.

        Raises:
            Exception: If there is no data to save.
        """
        # filepath = os.path.join(os.path.dirname(os.path.realpath('__file__')), filename)

        try:
            with open(filename, 'rb') as f:
                self.predecessors = pickle.load(f)
        except:
            self.predecessors = None
            print("No data to fetch")

    def save_pred_and_dist(self, filename="C:\\Users\\Sneez\\Desktop\\Snowplowing\\Snowplow-Routing-Middleton\\Snowplow-Routing-Middleton\\graph_data\\preds_and_dists.pickle"):
        """
        Saves the predecessors and distances to a pickle file.

        Args:
            filename (str): The path to save the pickle file.

        Raises:
            Exception: If there is no data to save. The predecessors and distances must be computed first.
        """
        # filepath = os.path.join(os.path.dirname(os.path.realpath('__file__')), filename)
        if self.predecessors is None:
            raise Exception("No data to save. First compute predecessors and distances")
        with open(filename, 'wb') as f:
            pickle.dump(self.predecessors, f)
        

    def compute_pred_and_dist(self):
        """
        Computes the predecessors and distances using the Floyd-Warshall algorithm.
        """
        self.predecessors = nx.floyd_warshall_predecessor_and_distance(self.G_DUAL, weight='weight')[0]

    def compute_dists_array(self):
        """
        Computes the distance matrix using the Floyd-Warshall algorithm.
        """
        self.dists_array = nx.floyd_warshall_numpy(self.G_DUAL, weight='weight')

    def get_dist(self, edge1: tuple[int, int, int], edge2: tuple[int, int, int]) -> float:
        """
        Returns the shortest distance between two edges of the graph.

        Args:
            edge1 (tuple[int, int, int]): The first edge.
            edge2 (tuple[int, int, int]): The second edge.

        Returns:
            float: The shortest distance between the two edges.
        """
        node1 = self.edge_index_dict[edge1]
        node2 = self.edge_index_dict[edge2]
        return self.dists_array[node1][node2]

    def compute_nearest_neighbors(self) -> dict:
        """
        Computes and returns the nearest neighbors for each node in the graph.
        
        Returns:
            dict: A dictionary where the keys are the nodes in the graph and the values are arrays of nearest neighbors.
        """
        index_edge_dict = {index: edge for index, edge in enumerate(self.G_DUAL.nodes)}

        nearest_neighbors = dict()
        for i in range(self.dists_array.shape[0]):
            nearest_neighbors[self.index_edge_dict[i]] = np.empty(self.dists_array.shape[1],dtype='i,i,i')
            sorted_indices = np.argsort(self.dists_array[i], axis=0)
            for j in range(len(sorted_indices)):
                correct_index = sorted_indices[j]
                nearest_neighbors[index_edge_dict[i]][j] = index_edge_dict[correct_index]
        return nearest_neighbors
    
    def get_shortest_path(self, edge1: tuple[int, int, int], edge2: tuple[int, int, int]) -> list:
        """
        Returns the shortest path between two edges of the primal graph.

        Args:
            edge1: The first edge.
            edge2: The second edge.

        Returns:
            list: The shortest path between the two edges.
        """
        if self.params_match_graph():
            return nx.reconstruct_path(edge1, edge2, self.predecessors)
        else:
            raise Exception("Graph and parameters don't match. Recompute paths")

if __name__ == "__main__":
    from main import G_DUAL
    sp = ShortestPaths(G_DUAL, load_data=False, save_data=True)
    print("Done")