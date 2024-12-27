"""
This module provides functions to convert edge-based graphs to node-based graphs. 
The node-based graphs are used to represent turn angles and travel costs in a street network.
Functions:
- create_dual_streets(G: nx.MultiDiGraph, DEPOT: int, depotSource: bool=True, sourceNodes: bool=False) -> nx.MultiDiGraph:
    This function works with geometry linestring objects to calculate turn angles.
- create_dual_toy(G: nx.MultiDiGraph, depotSource: bool=True, sourceNodes: bool=False) -> nx.MultiDiGraph:
    Uses x,y coords embedded in toy primal graph instead of lat, long of streets data.
"""
from turns import angle_between_points, angle_between_vectors
from costs import cost_of_dual_node
import networkx as nx
import params
def create_dual_streets(G: nx.MultiDiGraph, DEPOT: int, depotSource: bool=True, sourceNodes: bool=False) -> nx.MultiDiGraph:
    """
    Creates a dual graph based on the given input graph from real streets data. 
    This function works with geometry linestring
    objects to calculate turn angles.

    Parameters:
    - G: nx.MultiDiGraph
        The input graph on which the dual graph will be based.
    - DEPOT: int
        The depot node of the graph.
    - depotSource: bool, optional (default=True)
        Specifies whether to add a depot source node to the dual graph.
    - sourceNodes: bool, optional (default=False)
        Specifies whether to add source and target nodes for all nodes to the dual graph.

    Returns:
    - L: nx.MultiDiGraph
        The dual graph created based on the input graph.
    """
    L = nx.MultiDiGraph()
    # Create a graph specific edge function.
    for from_node in G.edges(keys=True, data=True):
        # from_node is: (u,v,key, attrb)
        L.add_node(from_node[:3])
        for to_node in G.edges(from_node[1], keys=True, data=True):
            L.add_edge(from_node[:3], to_node[:3])
            # self loop means a u-turn is needed
            if(to_node[0] == to_node[1]):
                L.edges[from_node[:3], to_node[:3], 0]['angle'] = 180
                continue
                
            # add angles
            testL1 = from_node[3]['geometry']
            testL2 = to_node[3]['geometry']

            p0 = testL1.coords[-2]
            p1 = testL1.coords[-1]
            p2 = testL2.coords[0]
            p3 = testL2.coords[1]

            
            points = [p0, p1, p2, p3]
            shared = [x for x in points if points.count(x) > 1]
            remaining = [e for e in points if e not in [shared[0]]]
            angle_value = angle_between_points(remaining[0], shared[0], remaining[1])
            # cost is turn cost + travel time of the first edge
            L.edges[from_node[:3], to_node[:3], 0]['weight'] = cost_of_dual_node(from_node, angle_value)
            
    if sourceNodes:
        # add source and target nodes
        for node in G:
            L.add_node(str(node) + "_source", weight=0)
        # add edges connecting source and targets to valid gateways
        for node in L.nodes():
            # skip the source nodes which are strings. The other nodes are tuples
            if (type(node) == str):
                continue
                
            # for a digraph, not a multigraph:
            # source -> first node
            # second node -> source
            L.add_edge(str(node[0]) + "_source", node, weight=0)
            L.add_edge(node, str(node[1]) + "_source", weight=0)

    if depotSource:
        print("Adding depot source", DEPOT)
        L.add_node((DEPOT,DEPOT,0))
        for edge in G.out_edges(DEPOT, keys=True, data=True):
            # instant travel from depot onto any road off the depot
            L.add_edge((DEPOT,DEPOT,0), edge[0:3], weight=0)
        for edge in G.in_edges(DEPOT, keys=True, data=True):
            # weight is nonzero to account for travel time of the last edge right before depot
            L.add_edge(edge[0:3], (DEPOT,DEPOT,0), weight=edge[3]['travel_time'])

    return L


def create_dual_toy(G: nx.MultiDiGraph, depotSource: bool=True, sourceNodes: bool=False) -> nx.MultiDiGraph:
    """
    Creates a dual graph based on the given toy input graph. Uses x and y coords 
    embedded in toy primal graph instead of lat, long of streets data
    

    Parameters:
    - G: nx.MultiDiGraph
        The input graph on which the dual graph will be based.
    - depotSource: bool, optional (default=True)
        Specifies whether to add a depot source node to the dual graph.
    - sourceNodes: bool, optional (default=False)
        Specifies whether to add source and target nodes for all nodes to the dual graph.

    Returns:
    - L: nx.MultiDiGraph
        The dual graph created based on the input graph.
    """
    DEPOT = params.DEPOT
    L = nx.MultiDiGraph()
    # Create a graph specific edge function.
    for from_node in G.edges(keys=True, data=True):
        # from_node is: (u,v,key, attrb)
        L.add_node(from_node[:3])
        for to_node in G.edges(from_node[1], keys=True, data=True):
            L.add_edge(from_node[:3], to_node[:3])
            edge1_dual, edge2_dual = from_node[:3], to_node[:3]
        
            first_node = edge1_dual[0]
            second_node = edge1_dual[1]
            third_node = edge2_dual[1]

            v_x = G.nodes[second_node]['x']-G.nodes[first_node]['x']
            v_y = G.nodes[second_node]['y']-G.nodes[first_node]['y']

            w_x = G.nodes[third_node]['x']-G.nodes[second_node]['x']
            w_y = G.nodes[third_node]['y']-G.nodes[second_node]['y']

            v = (v_x, v_y)
            w = (w_x, w_y)
            angle_value = angle_between_vectors(v,w)

            L.edges[from_node[:3], to_node[:3], 0]['weight'] = cost_of_dual_node(from_node, angle_value)
    
    if sourceNodes:
        # add source and target nodes
        for node in G:
            L.add_node(str(node) + "_source", weight=0)
        # add edges connecting source and targets to valid gateways
        for node in L.nodes():
            # skip the source nodes which are strings. The other nodes are tuples
            if (type(node) == str):
                continue
                
            # for a digraph, not a multigraph:
            # source -> first node
            # second node -> source
            L.add_edge(str(node[0]) + "_source", node, weight=0)
            L.add_edge(node, str(node[1]) + "_source", weight=0)

    if depotSource:
        L.add_node((DEPOT,DEPOT,0))
        for edge in G.edges((DEPOT,DEPOT,0), keys=True):
            L.add_edge((DEPOT,DEPOT,0), edge, weight=0)
        for edge in G.in_edges((DEPOT,DEPOT,0), keys=True, data=True):
            L.add_edge(edge[0:3], (DEPOT,DEPOT,0), weight=edge[3]['travel_time'])
    return L

