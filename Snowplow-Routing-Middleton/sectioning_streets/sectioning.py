import networkx as nx
import pandas as pd
import shapely
import osmnx as ox
import geopandas as gpd

import sys
import os
# getting the name of the directory
# where the this file is present.
current = os.path.dirname(os.path.realpath(__file__))
 
# Getting the parent directory name
# where the current directory is present.
parent = os.path.dirname(current)
 
# adding the parent directory to 
# the sys.path.
sys.path.append(parent)
 
# importing the module
from initialize import create_full_streets, config_graph_attributes

def config_sectioned_component(G: nx.MultiDiGraph) -> nx.MultiGraph:
    """
    Configures the sectioned component of the graph.
    Parameters:
    - G (nx.MultiDiGraph): The input graph.
    Returns:
    - nx.MultiGraph: The modified graph with the sectioned component configured.
    """
    
    scc = list(nx.strongly_connected_components(G)) # strongly connected components
    scc.remove(max(scc, key=len))

    for i in scc:
        for j in i:
            G.remove_node(j) # remove all but the strongest connected component from G
    
    G = nx.convert_node_labels_to_integers(G)
    config_graph_attributes(G)
    return G

def fill_missing_node_coords(G_sectioned: nx.MultiDiGraph, G_osm: nx.MultiDiGraph):
    """
    Fills missing node coordinates in a sectioned component of a graph.
    Parameters:
    - G_sectioned (nx.MultiDiGraph): The sectioned component of the graph.
    - G_osm (nx.MultiDiGraph): The original graph.
    """
    
    for node in G_sectioned.nodes(data=True):
        if 'x' not in node[1]:
            node_coords = G_osm.nodes[node[0]]['x'], G_osm.nodes[node[0]]['y']
            node[1]['x'] = node_coords[0]
            node[1]['y'] = node_coords[1]
    
def create_sectioned_component(G_full: nx.MultiDiGraph, nodes_full: gpd.GeoDataFrame, edges_full: gpd.GeoDataFrame, polygon: shapely.Polygon) -> nx.MultiDiGraph:
    """
    Create a sectioned component of a graph within a given polygon.
    
    Parameters:
    G_full (nx.MultiDiGraph): The full graph.
    edges_full (gpd.GeoDataFrame): A GeoDataFrame containing all edges of the graph.
    nodes_full (gpd.GeoDataFrame): A GeoDataFrame containing all nodes of the graph.
    polygon (shapely.Polygon): The polygon defining the area of interest.
    
    Returns:
    G_sectioned (nx.MultiDiGraph): A sectioned component of the graph within the given polygon.
    """

    edges_in_polygon = edges_full[edges_full.intersects(polygon)]
    nodes_in_polygon = nodes_full[nodes_full.intersects(polygon)]
    G_sectioned = ox.graph_from_gdfs(nodes_in_polygon, edges_in_polygon)

    fill_missing_node_coords(G_sectioned, G_full)

    return G_sectioned

def get_full_streets_nodes_edges() -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Retrieves the nodes and edges of a graph representing full streets.
    Returns:
        nodes (GeoDataFrame): A GeoDataFrame containing the nodes of the graph.
        edges (GeoDataFrame): A GeoDataFrame containing the edges of the graph.
    """
    
    G = create_full_streets()
    nodes, edges = ox.graph_to_gdfs(G)
    return nodes, edges, G

def load_polygon(path: str) -> shapely.Polygon:
    """
    Loads a polygon from a given file path.
    Parameters:
    - path (str): The path to the file containing the polygon.
    Returns:
    - polygon (shapely.Polygon): The polygon loaded from the file.
    """
    
    polygon = gpd.read_file(path)
    return polygon.geometry[0]


def section_component(polygon_path: str) -> nx.MultiDiGraph:
    """
    Sections a component of the full streets graph within a given polygon.
    Parameters:
    - polygon_path (str): The path to the file containing the polygon.
    Returns:
    - G_sectioned (nx.MultiDiGraph): A sectioned component of the graph within the given polygon.
    """
    
    nodes, edges, G_full = get_full_streets_nodes_edges()
    polygon = load_polygon(polygon_path)
    G_sectioned = create_sectioned_component(G_full, nodes, edges, polygon)
    G_sectioned = config_sectioned_component(G_sectioned)
    return G_sectioned