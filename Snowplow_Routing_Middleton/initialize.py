"""
This module provides functions to initialize and manipulate graphs for snowplow routing in Middleton.
Functions:
    create_small_toy() -> nx.MultiDiGraph:
        Creates a small toy graph for snowplow routing in Middleton.
    create_small_streets() -> nx.MultiDiGraph:
        Creates a graph instance for the small streets network graph.
    create_full_streets() -> nx.MultiDiGraph:
        Creates a full streets network graph.
"""

import pandas as pd
import osmnx as ox
import numpy as np
import geopandas as gpd
import networkx as nx
import pickle
import os
from .params import PLOW_SPEED_HIGHWAY, PLOW_SPEED_RESIDENTIAL, HIGH_PRIORITY_ROADS

def add_toy_street_info(G: nx.Graph) -> None:
    """
    Adds street information to the graph, 
    including road priorities, number of passes required, and salt requirements for each 
    road segment. To simulate real data, a random subset of roads are made unrequired and marked.

    Args:
        G (nx.Graph): The graph to which street information will be added.

    Returns:
        None
    """

    priority_keys = {"residential":1, "tertiary":2, "motorway_link":3, "primary":4}
    passes_keys = {"residential":1, "tertiary":1, "motorway_link":2, "primary":2}
    salt_keys = {"residential":1, "tertiary":1, "motorway_link":2, "primary":3}

    for edge in G.edges(data=True, keys=True):
        attrb = edge[3]
        highway_type = attrb['highway']
        attrb['priority'] = priority_keys[highway_type]
        if 'lanes' in attrb:
            if type(attrb['lanes']) == list:
                attrb['passes_rem'] = float(attrb['lanes'][0]) // 2
                attrb['salt_per'] = float(attrb['lanes'][0]) // 2
            else:
                attrb['passes_rem'] = float(attrb['lanes']) // 2
                attrb['salt_per'] = float(attrb['lanes']) // 2
        else:
            attrb['passes_rem'] = passes_keys[highway_type]
            attrb['salt_per'] = salt_keys[highway_type]
        attrb['serviced'] = False
    for edge in G.edges(data=True):
        attrb = edge[2]
        if np.random.random() < 0.1:
            attrb['passes_rem'] = 0
            attrb['priority'] = 0
            attrb['salt_per'] = 0
            attrb['serviced'] = True

def add_node_weighted_degree(G: nx.Graph) -> None:
    """
    Configures the graph attributes by calculating the weighted degree for each node
    and adding a new attribute called "deadheading_passes" to each edge.

    Parameters:
    G (nx.Graph): The input graph.

    Returns:
    None
    """

    for node in G.nodes:
        weight_deg = 0
        for edge in G.edges([node], data=True):
            weight_deg += edge[2]['passes_rem'] * edge[2]['priority']
        G.nodes[node]['weighted_degree'] = weight_deg

    # add a new attribute to each edge, called "deadheading_passes" and initially set to 0
    nx.set_edge_attributes(G, 0, "deadheading_passes")

def set_high_priority_roads(G: nx.MultiDiGraph) -> None:
    """
    Updates the priority attribute of the road network based on the information about highest priority streets
    provided by the city.

    Args:
        G (nx.MultiDiGraph): graph representing the road network
    Returns:
        None
    """
    for edge in G.edges(data=True):
        attrb = edge[2]
        name_of_edge = attrb.get("name", "None") # potentially have either 1 or 2 names depending on the linestring
        if type(name_of_edge) == str:
            name_of_edge = [name_of_edge]
        for name in name_of_edge:
            # special cases based on configuration set by params file
            if name == "Century Avenue":
                lstring = attrb.get("geometry")
                for long, lat in lstring.coords:
                    if long < -89.509817:
                        attrb.update({"priority": 10})
                        break
            elif name == "Park Street:":
                lstring = attrb.get("geometry")  
                for long, lat in lstring.coords:
                    if lat < 43.097088:
                        attrb.update({"priority": 10})
                        break
            elif name in HIGH_PRIORITY_ROADS:
                attrb.update({"priority": 10})
                break

def create_small_toy() -> nx.MultiDiGraph:
    """
    Create a small toy graph for snowplow routing in Middleton.

    Reads the edge and node data from CSV files and constructs a directed graph.
    Calculates the weighted degree of each node based on the edges and their attributes.
    Sets the weighted degree as a node attribute.
    Sets the deadheading_passes attribute of all edges to 0.
    Sets all edges to initially be unserviced

    Returns:
        G (networkx.MultiDiGraph): The constructed graph.
    """
    edgeFile = os.path.dirname(__file__) + "/graph_data/edges.csv"
    nodeFile = os.path.dirname(__file__) + "/graph_data/nodes.csv"
    edgelist = pd.read_csv(edgeFile)
    nodelist = pd.read_csv(nodeFile)

    G = nx.MultiDiGraph()
    for i, edges in edgelist.iterrows():
        G.add_edge(edges.iloc[0], edges.iloc[1], **edges.iloc[2:].to_dict())
        G.add_edge(edges.iloc[1], edges.iloc[0], **edges.iloc[2:].to_dict())

    for i, nodes in nodelist.iterrows():
        attributes = nodes[1:].to_dict()
        weight_deg = 0
        for edge in G.edges([nodes['id']], data=True):
            weight_deg += edge[2]['passes_rem'] * edge[2]['priority']

        attributes["weighted_degree"] = weight_deg
        nx.set_node_attributes(G, {nodes['id']: attributes})

    nx.set_edge_attributes(G, 0, "deadheading_passes")
    nx.set_edge_attributes(G, False, "serviced")

    return G

def create_small_streets() -> nx.MultiDiGraph:
    """
    Creates a graph instance for the small streets network graph.

    Returns:
        nx.MultiDiGraph: The small streets network graph.
    """
    point = (43.095273490280036, -89.50918121778929)
    G = ox.graph_from_point(point, dist=500, network_type='drive')
    G = ox.add_edge_speeds(G)
    G = ox.add_edge_travel_times(G)

    G = nx.convert_node_labels_to_integers(G)

    scc = list(nx.strongly_connected_components(G)) # strongly connected components
    scc.remove(max(scc, key=len))

    for i in scc:
        for j in i:
            G.remove_node(j) # remove all but the strongest connected component from G

    add_toy_street_info(G)
    set_high_priority_roads(G)
    add_node_weighted_degree(G)
    # add geometry attribute to all edges
    nodes, edges = ox.graph_to_gdfs(G)
    G = ox.graph_from_gdfs(nodes, edges)
    return G

def get_salt_from_length(length: float) -> float:
    """
    Returns the amount of salt (in lbs) required for a given road length in meters. 250 lbs/mile

    Parameters:
        length (float): The length of the road segment.

    Returns:
        float: The amount of salt required for the road segment.
    """
    length_mile = length * 0.000621371
    return length_mile * 250

def is_culdesac(G: nx.MultiDiGraph, node: int) -> bool:
    """
    Determines if a given node in a MultiDiGraph represents a cul-de-sac.
    Parameters:
        G (nx.MultiDiGraph): The MultiDiGraph representing the road network.
        node (int): The node to check.
    Returns:
        bool: True if the node is a cul-de-sac, False otherwise.
    """

    if G.out_degree(node) == 0:
        return True
    
    edge = list(G.edges(node, data=True))
    attrb = edge[0][2]

    # if 'roadtype' in attrb:
    #     if attrb['roadtype'] == 'Ct' or attrb['roadtype'] == 'Cir':
    #         if attrb['highway'] == 'residential':
    #             return True

    return G.out_degree(node) == 1 and G.in_degree(node) == 1 and attrb['highway'] == 'residential' and attrb['reversed'] == True and attrb['length'] < 500

passes_keys = {"motorway_link":1, "tertiary_link":1, "secondary_link":1, "primary_link":1, "unclassified":1, "residential":2, "tertiary":2, "secondary":3, "primary":3, "motorway":6}
small_roads = ["motorway_link", "tertiary_link", "secondary_link", "primary_link", "unclassified"]
def calc_passes(oneway: bool, width: float, highway: str, roadtype: str) -> int:
    """
    Calculates the number of passes required for a given road segment.

    Parameters:
        oneway (bool): Whether the road segment is one-way.
        width (float): The width of the road segment.
        highway (str): The type of the road segment.

    Returns:
        int: The number of passes required for the road segment.
    """
    if highway in small_roads:
        return 1
    if np.isnan(width) and roadtype != "Blvd":
        return passes_keys[highway]//2 if oneway else passes_keys[highway]
    
    if roadtype == "Blvd":
        return 2 if oneway else 4

    if width <= 36:
        return 1 if oneway else 2
    else:
        return 1 if oneway else 3
    
def create_full_streets() -> nx.MultiDiGraph:
    """
    Creates a full streets network graph

    Returns:
        nx.MultiDiGraph: The full scale streets network
    """
    # Read the shapefile
    osm_data_filepath = os.path.dirname(__file__) + "/graph_data/OSMWithData.gpkg"

    street_gdf = gpd.read_file(osm_data_filepath)

    osm_graph_filepath = os.path.dirname(__file__) + "/graph_data/streets_graph.pickle"
    # Read the OSM Graph
    G = pickle.load(open(osm_graph_filepath, 'rb'))
    G = nx.convert_node_labels_to_integers(G)

    nodes, edges = ox.graph_to_gdfs(G) # better than momepy b/c fills in missing geometry attributes

    edges['jurisdiction'] = np.array(street_gdf['Jurisdicti'])
    edges['width'] = np.array(street_gdf['With_EE_ft'])
    edges['roadtype'] = np.array(street_gdf['abvPostTyp'])
    edges['maintainer'] = np.array(street_gdf['Maintained'])
    G = ox.graph_from_gdfs(nodes, edges)
    priority_keys = {"motorway_link":1, "tertiary_link":1, "secondary_link":1, "primary_link":1, "unclassified":1, "residential":2, "tertiary":3, "secondary":4, "primary":5, "motorway":6}

    priorities = np.empty(len(edges))
    passes = np.empty(len(edges))
    salt = np.empty(len(edges))
    serviced = np.empty(len(edges), dtype=bool)
    culdesac = np.empty(len(edges), dtype=bool)
    plow_time = np.empty(len(edges))
    # go through each edge and update dictionary
    for index, edges_data in enumerate(edges.iterrows()):
        edge = edges_data[0]
        data = edges_data[1]

        highway_type = data['highway']
        length_meters = data['length']
        width = data['width']
        roadtype = data['roadtype']
        oneway = data['oneway']

        if street_gdf.iloc[index]['Jurisdicti'] == "City":
            priorities[index] = priority_keys[highway_type]
            passes[index] = calc_passes(oneway, width, highway_type, roadtype)
            salt[index] = get_salt_from_length(length_meters)
            serviced[index] = False
        else:
            priorities[index] = 0
            passes[index] = 0
            salt[index] = 0
            serviced[index] = True
        
        if highway_type == "residential":
            plow_time[index] = length_meters / PLOW_SPEED_RESIDENTIAL
        else:
            plow_time[index] = length_meters / PLOW_SPEED_HIGHWAY
        culdesac[index] = is_culdesac(G, edge[1]) # edge[1] corresponds to the second node of the edge
    edges['priority'] = priorities
    edges['passes_rem'] = passes
    edges['salt_per'] = salt
    edges['serviced'] = serviced
    edges['culdesac'] = culdesac
    edges['travel_time'] = plow_time

    G = ox.graph_from_gdfs(nodes, edges)
    for edge in G.edges(data=True, keys=True):
        if 'roadtype' in edge[3]:
            if (edge[3]['roadtype'] == "Ct" or edge[3]['roadtype'] == "Cir") and edge[3]['highway'] == 'residential':
                edge[3]['culdesac'] = True
        elif "name" in edge[3]:
            if edge[3]['name'] == "Bunker Hill Lane" or edge[3]['name'] == "Patrick Henry Way":
                edge[3]['culdesac'] = True
    scc = list(nx.strongly_connected_components(G)) # strongly connected components
    scc.remove(max(scc, key=len))

    for i in scc:
        for j in i:
            G.remove_node(j) # remove all but the strongest connected component from G
    set_high_priority_roads(G)
    add_node_weighted_degree(G)

    return G
def add_multi_edges(G: nx.Graph) -> nx.MultiDiGraph:
    """
    Adds multiple edges to a graph based on the 'passes_rem' attribute of each edge.

    Parameters:
        G (nx.Graph): The input graph.

    Returns:
        nx.MultiDiGraph: The modified graph with multiple edges.

    """
    G_new = nx.MultiDiGraph(G)
    for edge in G.edges(data=True):
        attrbs = edge[2]
        for i in range(int(attrbs['passes_rem']-1)):
            G_new.add_edge(edge[0], edge[1],i+1)
            nx.set_edge_attributes(G_new, {(edge[0], edge[1],i+1):attrbs})
    return G_new

if __name__ == "__main__":
    create_small_toy()
    create_full_streets()
    create_small_streets()
    