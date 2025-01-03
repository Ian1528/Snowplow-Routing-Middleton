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
from .params import find_depot
import folium
import folium.plugins
import datetime

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

def plot_moving_routes_folium(G: nx.MultiDiGraph, full_route: list[tuple[int, int, int]], m: folium.Map | None, label_color: str, path_color: str, dif=1e-4) -> folium.Map:
    """
    Plots moving routes on an animated Folium map.

    Args:
        G (nx.MultiDiGraph): The graph containing the route data.
        full_route (list[tuple[int, int, int]]): A list of tuples representing the edges in the route.
        m (folium.Map | None): An existing Folium map to plot on, or None to create a new map.
        label_color (str): The color of the labels for the markers.
        path_color (str): The color of the path lines.
        dif (float, optional): The difference in latitude or longitude at which to add intermediate points to smooth the animation. Defaults to 1e-4.
    Returns:
        folium.Map: The Folium map with the plotted routes.
    """
    if m is None:
        m = folium.Map(location=[43.1, -89.5], zoom_start=12)
    count = 0
    current_time = datetime.datetime.now()
    features = list()
    for edge in full_route:
        edge_data = G.get_edge_data(edge[0], edge[1], edge[2])
        if edge_data is not None:
            name = edge_data.get("name", "Unnamed")   
            lstring = edge_data['geometry']
            lat_long_coords = [(y, x) for x, y in lstring.coords]
            lstring_lengthed_coords = list()
            for i in range(len(lat_long_coords)):
                lat,long=lat_long_coords[i]
                if i < len(lat_long_coords) - 1:
                    next_lat, next_long = lat_long_coords[i + 1]
                    dif_lat, dif_long = next_lat - lat, next_long - long

                    if dif_lat > dif or dif_long > dif:
                        num_intervals = int(max(abs(dif_lat), abs(dif_long)) // dif)
                        new_lat_coords = np.linspace(lat, next_lat, num_intervals)
                        new_long_coords = np.linspace(long, next_long, num_intervals)

                        lstring_lengthed_coords.extend([(new_long_coords[i], new_lat_coords[i]) for i in range(num_intervals)])
                    else:
                        lstring_lengthed_coords.append((long, lat))
                else:
                    lstring_lengthed_coords.append((long, lat))
            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": lstring_lengthed_coords
                },
                "properties": {
                    "times": [str(current_time + datetime.timedelta(minutes=i)) for i in range(len(lstring_lengthed_coords))],
                    "name": name,
                    "edge": edge,
                    "order": count,
                    "style": {
                        "color": path_color,
                        "weight": 3.5
                    },
                }
            }
            features.append(feature)
            folium.PolyLine(locations=lat_long_coords, color="black", weight=1, tooltip=edge_data).add_to(m)
            current_time += datetime.timedelta(minutes=len(lstring_lengthed_coords))
            
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