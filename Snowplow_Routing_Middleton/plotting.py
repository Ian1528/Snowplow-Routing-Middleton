"""
This module provides functions for plotting graphs and routes using Matplotlib and Folium.
Functions:
    draw_labeled_multigraph(G, attr_name=None, ax=None, color='blue', title="Graph Representation of Town", size=(100,75), plotDepot=False, depotCoords=None):
        Draws a labeled multigraph using Matplotlib.
    add_order_attribute(G, routes):
        Adds an 'order' attribute to the edges of the graph G based on the given routes.
    plot_routes_folium(G: nx.MultiDiGraph, full_route: list[tuple[int, int, int]], m: folium.Map | None, label_color: str, path_color: str):
        Plots routes on a Folium map with labeled edges.
    plot_moving_routes_folium(G: nx.MultiDiGraph, full_route: list[tuple[int, int, int]], m: folium.Map | None, label_color: str, path_color: str):
        Plots moving routes on a Folium map with labeled edges and timestamps.
"""

import itertools as it
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from .params import find_depot
import folium
import folium.plugins
import datetime
import shapely

def get_node_pos(G: nx.MultiDiGraph) -> dict:
    """
    Get the position of each node in the graph.

    Args:
        G (nx.MultiDiGraph): the graph representing the street network

    Returns:
        dict: a dictionary containing the position of each node in the graph
    """
    return {node[0]: (node[1]['x'], node[1]['y']) for node in G.nodes(data=True)}


def draw_labeled_multigraph(G, attr_name=None, ax=None, color='blue', title="Graph Representation of Town", size=(100,75), plotDepot=False):
    """
    Draws a labeled multigraph using NetworkX and Matplotlib.
    
    Args:
        G (networkx.Graph): The graph to be drawn.
        attr_name (str, optional): The edge attribute to be displayed as labels. Defaults to None.
        ax (matplotlib.axes.Axes, optional): The axes on which to draw the graph. Defaults to None.
        color (str, optional): The color of the edge labels. Defaults to 'blue'.
        title (str, optional): The title of the plot. Defaults to "Graph Representation of Town".
        size (tuple, optional): The size of the plot. Defaults to (100, 75).
        plotDepot (bool, optional): Whether to plot the depot node. Defaults to False.
    
    Returns:
        None
    """
    
    DEPOT, DEPOTX, DEPOTY  = find_depot(G)
    pos = get_node_pos(G)
    # Works with arc3 and angle3 connectionstyles
    connectionstyle = [f"arc3,rad={r}" for r in it.accumulate([0.15] * 4)]
    # connectionstyle = [f"angle3,angleA={r}" for r in it.accumulate([30] * 4)]

    node_lables = dict([(node, "") if node != DEPOT else (node, "depot") for node in G.nodes()]) # label the depot

    plt.figure(figsize=size)
    plt.title(title, size=50)

    if plotDepot:
        plt.plot(DEPOTX, DEPOTY, 'ro', label='Depot', markersize=50)

    nx.draw_networkx_nodes(G, pos, ax=ax, node_color="black")
    nx.draw_networkx_labels(G, pos, font_size=50, ax=ax, labels=node_lables)
    nx.draw_networkx_edges(
        G, pos, edge_color="grey", connectionstyle=connectionstyle, ax=ax, arrows=True, arrowsize=25
    )
    if attr_name is not None:
        labels = {
            tuple(edge): f"{attrs[attr_name]}"
            for *edge, attrs in G.edges(keys=True, data=True)
        }
        nx.draw_networkx_edge_labels(
            G,
            pos,
            labels,
            connectionstyle=connectionstyle,
            label_pos=0.5,
            font_color=color,
            font_size=25,
            bbox={"alpha": 0},
            ax=ax,
        )
    plt.show()

def add_order_attribute(G: nx.MultiDiGraph, routes: list[list[tuple[int, int, int]]]) -> nx.MultiDiGraph:
    """
    Adds an 'order' attribute to the edges of the graph G based on the given routes.

    Args:
        G (nx.MultiDiGraph): The graph to which the 'order' attribute will be added.
        routes (list[list[tuple[int, int, int]]]): A list of routes, where each route is a list of edges.

    Returns:
        nx.MultiDiGraph: The graph with the 'order' attribute added to its edges.
    """
    G_graph = G.copy()
    count = 0
    for route in routes:
        for edge in route:
            if G_graph[edge[0]][edge[1]][edge[2]].get('order') is None:
                G_graph[edge[0]][edge[1]][edge[2]]['order'] = str(count)
            else:
                G_graph[edge[0]][edge[1]][edge[2]]['order'] += ", " + str(count)
            count += 1
    for edge in G_graph.edges(data=True):
        if edge[2].get('order') is None:
            if edge[2].get('priority') == 0:
                edge[2]['order'] = "N/A"  
            else:
                edge[2]['order'] = "UNSERVICED"
    return G_graph

def plot_routes_folium(G: nx.MultiDiGraph, full_route: list[tuple[int, int, int]], m: folium.Map | None, label_color: str, path_color: str) -> folium.Map:
    """
    Plots routes on a static Folium map.
    Args:
        G (nx.MultiDiGraph): The graph containing the route data.
        full_route (list[tuple[int, int, int]]): A list of tuples representing the edges in the route.
        m (folium.Map | None): An existing Folium map to plot on, or None to create a new map.
        label_color (str): The color of the labels for the markers.
        path_color (str): The color of the path lines.
    Returns:
        folium.Map: The Folium map with the plotted routes.
    """
    
    if m is None:
        m = folium.Map(location=[43.1, -89.5], zoom_start=12)
    count = 0
    for i, edge in enumerate(full_route):
        edge_data = G.get_edge_data(edge[0], edge[1], edge[2])
            
        if edge_data is not None:
            plot_marker = True
            name = edge_data.get("name", "Unnamed")   
            if i < len(full_route)-1:
                edge_data_next = G.get_edge_data(full_route[i+1][0], full_route[i+1][1], full_route[i+1][2])
                if edge_data_next is not None and "name" in edge_data_next and "name" in edge_data:
                    if edge_data_next["name"] == edge_data["name"]:
                        plot_marker = False
            lstring = edge_data['geometry']
            # swap long lat to lat long
            lstring = lstring.__class__([(y, x) for x, y in lstring.coords])
            midpoint = len(list(lstring.coords))//2
            icon_number = folium.plugins.BeautifyIcon(
                border_color=label_color,
                border_width=1,
                text_color=label_color,
                number=count,
                inner_icon_style="margin-top:2;",
            )
            folium.PolyLine(locations=lstring.coords, color=path_color, weight=1, tooltip=edge_data).add_to(m)
            if plot_marker:
                folium.Marker(location=lstring.coords[midpoint], popup=f"Edge {count}: {name}", icon=icon_number).add_to(m)
            count += 1
    return m


def find_clusters_of_edges(G: nx.MultiDiGraph) -> tuple[dict[tuple[list[int, int], list[tuple[int,int,int]]]], dict[tuple[tuple[int, int, int], tuple[int,int,int]]]]:

    """
    Find pairs of edges in a graph that represent the same road segment in opposite directions, and find 
    pairs of edges that represent the same road segment in the same direction.

    Args:
        G (nx.MultiDiGraph): The graph representing the road network.
    Returns:
        tuple[dict[tuple[list[int, int], list[tuple[int,int,int]]]], dict[tuple[tuple[int, int, int], tuple[int,int,int]]]]:
        A tuple of two dictionaries. 
        The first dictionary maps linestrings to the edges that correspond to it.
        and the second dictionary contains pairs of edges that represent the same road segment in opposite directions
    """
    lstrings_to_single_edges = dict() # dictionary mapping linestrings to edges
    antipairs_dict = dict()
    lstrings_to_multi_edges = dict()
    for edge in G.edges(data=True, keys=True):
        edge_tup = (edge[0], edge[1], edge[2])
        if edge[3]['geometry'] is not None:
            lstring = tuple(edge[3]['geometry'].coords)
            # matching antipair (same road, opposite direction)
            if lstring[::-1] in lstrings_to_single_edges.keys():
                antipairs_dict.update({edge_tup: lstrings_to_single_edges[lstring[::-1]]})
                antipairs_dict.update({lstrings_to_single_edges[lstring[::-1]]: edge_tup})
            
            lstrings_to_single_edges.update({lstring: edge_tup})
            lstrings_to_multi_edges.update({lstring: lstrings_to_multi_edges.get(lstring, []) + [edge_tup]})
    return lstrings_to_multi_edges, antipairs_dict
def lengthen_lstring_coords(lstring: shapely.LineString, diff: float) -> list[tuple[float, float]]:
    """
    Get the coordinates of a LineString with additional points added to make the lines smoother.

    Helper function for plot_moving_routes_folium.

    Args:
        lstring (shapely.LineString): The LineString to get coordinates from.
        diff (float): The maximum difference between two points in the LineString.
    Returns:
        list[tuple[float, float]]: A list of coordinates for the LineString.
    """
    lat_long_coords = [(y, x) for x, y in lstring.coords]
    lstring_lengthed_coords = list()
    for i in range(len(lat_long_coords)):
        lat,long=lat_long_coords[i]
        if i < len(lat_long_coords) - 1:
            next_lat, next_long = lat_long_coords[i + 1]
            dif_lat, dif_long = next_lat - lat, next_long - long

            if dif_lat > diff or dif_long > diff:
                num_intervals = int(max(abs(dif_lat), abs(dif_long)) // diff)
                new_lat_coords = np.linspace(lat, next_lat, num_intervals)
                new_long_coords = np.linspace(long, next_long, num_intervals)

                lstring_lengthed_coords.extend([(new_long_coords[i], new_lat_coords[i]) for i in range(num_intervals)])
            else:
                lstring_lengthed_coords.append((long, lat))
        else:
            lstring_lengthed_coords.append((long, lat))
    return lstring_lengthed_coords, lat_long_coords

def plot_moving_routes_folium(G: nx.MultiDiGraph, full_route: list[tuple[int, int, int]], m: folium.Map | None, label_color: str, path_color: str, dif=1e-4) -> folium.Map:
    """
    Plots moving routes on an animated Folium map.

    Args:
        G (nx.MultiDiGraph): The graph containing the route data.
        full_route (list[tuple[int, int, int]]): A list of tuples representing the edges in the route.
        m (folium.Map | None): An existing Folium map to plot on, or None to create a new map.
        label_color (str): The color of the labels for the markers.
        path_color (str): The color of the path lines.
    Returns:
        folium.Map: The Folium map with the plotted routes.
    """
    G_copy = G.copy()
    if m is None:
        m = folium.Map(location=[43.1, -89.5], zoom_start=12)
    count = 0
    current_time = datetime.datetime.now()
    features = list()

    lstrings_to_multi_edges, antipairs_dict = find_clusters_of_edges(G)
    # find the maximum number of passes an edge requires based on edge attributes in G
    mapped_edges, partially_mapped_antipairs = set(), set()
    for edge in full_route:
        edge_data = G_copy.get_edge_data(edge[0], edge[1], edge[2])   
        if edge_data is None:
            continue
        
        graph_attributes = {"color": path_color, "dashed": False, "weight": 5}
        # if the edge is a part of a pair
        lstring = tuple(edge_data['geometry'].coords)
        # set the weight based on the number of times this linsestring has been serviced
        num_mapped_edges = len([pair_edge for pair_edge in lstrings_to_multi_edges[lstring] if pair_edge in mapped_edges])+1
        total_edges = len(lstrings_to_multi_edges[lstring])+1
        graph_attributes['weight']= num_mapped_edges/total_edges * 5

        if edge in antipairs_dict.keys():
            # if the first part of the edge hasn't been serviced, make the line dashed
            if edge not in partially_mapped_antipairs and antipairs_dict[edge] not in partially_mapped_antipairs:
                graph_attributes['dashed'] = True
                partially_mapped_antipairs.add(edge)

            
        deadhead = False if edge not in mapped_edges else True
        mapped_edges.add(edge)

        name = edge_data.get("name", "Unnamed")
        if deadhead:
            graph_attributes['color'] = "red"

        lstring = edge_data['geometry']
        coords, lat_long_coords = lengthen_lstring_coords(lstring, dif)
        feature = {
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": coords
            },
            "properties": {
                "times": [str(current_time + datetime.timedelta(minutes=i)) for i in range(len(coords))],
                "name": name,
                "edge": edge,
                "order": count,
                "style": {
                    "color": graph_attributes["color"],
                    "weight": graph_attributes['weight'],
                    "dashArray": "5, 10" if graph_attributes['dashed'] else None
                },
            }
        }
        features.append(feature)
        folium.PolyLine(locations=lat_long_coords, color="black", weight=.5, tooltip=str(edge) + " passes: " + str(edge_data.get("passes_req"))).add_to(m)
        current_time += datetime.timedelta(minutes=len(coords))
    folium.plugins.TimestampedGeoJson(
        {
            "type": "FeatureCollection",
            "features": features,
        },
        period="PT5M",
        add_last_point=False,
    ).add_to(m)
    folium.plugins.TimestampedGeoJson(
        {
            "type": "FeatureCollection",
            "features": features,
        },
        period="PT5M",
        add_last_point=True,
        duration="PT1M"
    ).add_to(m)
    return m