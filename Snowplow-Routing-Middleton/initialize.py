import pandas as pd
import osmnx as ox
import numpy as np
import geopandas as gpd
import networkx as nx
import pickle

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

def config_graph_attributes(G: nx.Graph) -> None:
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


def create_small_toy(edgeFile="Snowplow-Routing-Middleton/graph_data/edges.csv", nodeFile="Snowplow-Routing-Middleton/graph_data/nodes.csv") -> nx.MultiDiGraph:
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
    Creates a small streets network graph.

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
    config_graph_attributes(G)

    # add geometry attribute to all edges
    nodes, edges = ox.graph_to_gdfs(G)
    G = ox.graph_from_gdfs(nodes, edges)
    return G

def create_full_streets() -> nx.MultiDiGraph:
    """
    Creates a full streets network graph

    Returns:
        nx.MultiDiGraph: The full scale streets network
    """
    # Read the shapefile
    street_gdf = gpd.read_file("C:\\Users\\Sneez\\Desktop\\Snowplowing\\Snowplow-Routing-Middleton\\Snowplow-Routing-Middleton\\graph_data\\OSMWithData.gpkg")

    # Read the OSM Graph
    G = pickle.load(open("C:\\Users\\Sneez\\Desktop\\Snowplowing\\Snowplow-Routing-Middleton\\Snowplow-Routing-Middleton\\graph_data\\streets_graph.pickle", 'rb'))
    G = nx.convert_node_labels_to_integers(G)

    nodes, edges = ox.graph_to_gdfs(G) # better than momepy b/c fills in missing geometry attributes

    edges['jurisdiction'] = np.array(street_gdf['Jurisdicti'])
    edges['width'] = np.array(street_gdf['With_EE_ft'])
    edges['roadtype'] = np.array(street_gdf['abvPostTyp'])
    edges['maintainer'] = np.array(street_gdf['Maintained'])

    priority_keys = {"motorway_link":1, "tertiary_link":1, "secondary_link":1, "primary_link":1, "unclassified":1, "residential":2, "tertiary":3, "secondary":4, "primary":5, "motorway":6}
    passes_keys = {"motorway_link":1, "tertiary_link":1, "secondary_link":1, "primary_link":1, "unclassified":1, "residential":2, "tertiary":3, "secondary":4, "primary":5, "motorway":6}
    salt_keys = {"motorway_link":1, "tertiary_link":1, "secondary_link":1, "primary_link":1, "unclassified":1, "residential":2, "tertiary":3, "secondary":4, "primary":5, "motorway":6}

    priorities = np.empty(len(edges))
    passes = np.empty(len(edges))
    salt = np.empty(len(edges))
    serviced = np.empty(len(edges), dtype=bool)

    # go through each edge and update dictionary
    for row in range(len(edges)):
        highway_type = edges.iloc[row]['highway']
        if street_gdf.iloc[row]['Jurisdicti'] == "City":
            priorities[row] = priority_keys[highway_type]
            passes[row] = passes_keys[highway_type]
            salt[row] = salt_keys[highway_type]
            serviced[row] = False
        else:
            priorities[row] = 0
            passes[row] = 0
            salt[row] = 0
            serviced[row] = True


    edges['priority'] = priorities
    edges['passes_rem'] = passes
    edges['salt_per'] = salt
    edges['serviced'] = serviced
    
    G = ox.graph_from_gdfs(nodes, edges)
    scc = list(nx.strongly_connected_components(G)) # strongly connected components
    scc.remove(max(scc, key=len))

    for i in scc:
        for j in i:
            G.remove_node(j) # remove all but the strongest connected component from G

    config_graph_attributes(G)
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
    