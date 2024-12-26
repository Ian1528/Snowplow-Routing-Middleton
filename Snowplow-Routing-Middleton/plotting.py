import itertools as it
import matplotlib.pyplot as plt
import networkx as nx
from params import find_depot
import folium
import folium.plugins
import datetime

def get_node_pos(G):
    return {node[0]: (node[1]['x'], node[1]['y']) for node in G.nodes(data=True)}


def draw_labeled_multigraph(G, attr_name=None, ax=None, color='blue', title="Graph Representation of Town", size=(100,75), plotDepot=False, depotCoords=None):
    """_summary_

    Args:
        G (_type_): _description_
        pos (_type_): _description_
        attr_name (_type_, optional): _description_. Defaults to None.
        ax (_type_, optional): _description_. Defaults to None.
        color (str, optional): _description_. Defaults to 'blue'.
        title (str, optional): _description_. Defaults to "Graph Representation of Town".
        plotDepot (bool, optional): _description_. Defaults to False.
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

def add_order_attribute(G, routes):
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

def add_order_attribute_from_edges(G, routes):
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


def plot_routes_folium(G: nx.MultiDiGraph, full_route: list[tuple[int, int, int]], m: folium.Map | None, label_color: str, path_color: str):
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

def plot_moving_routes_folium(G: nx.MultiDiGraph, full_route: list[tuple[int, int, int]], m: folium.Map | None, label_color: str, path_color: str):
    if m is None:
        m = folium.Map(location=[43.1, -89.5], zoom_start=12)
    count = 0
    current_time = datetime.datetime.now()
    features = list()
    point_features = list()
    for i, edge in enumerate(full_route):
        edge_data = G.get_edge_data(edge[0], edge[1], edge[2])
        current_time = current_time + datetime.timedelta(minutes=1)#.split('.')[0]
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
            # lstring_lat_long = lstring.__class__([(y, x) for x, y in lstring.coords])
            lat_long_coords = [(y, x) for x, y in lstring.coords]
            midpoint = len(list(lstring.coords))//2
            icon_number = folium.plugins.BeautifyIcon(
                border_color=label_color,
                border_width=1,
                text_color=label_color,
                number=count,
                inner_icon_style="margin-top:2;",
            )
            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": list(lstring.coords)
                },
                "properties": {
                    "times": [str(current_time) for i in range(len(list(lstring.coords)))],
                    "name": name,
                    "edge": edge,
                    "order": count,
                    "style": {
                        "color": path_color,
                        "weight": 3.5
                    },
                }
            }
            point_feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": list(lstring.coords)[-1]
                },
                "properties": {
                    "times": [str(current_time)],
                    "popup": f"Edge {count}: {name}",
                    "icon": "circle",
                    "iconstyle":{
                        "fillColor": "red",
                        "fillOpacity": 1,
                        "radius": 5,
                    },
                    "style": {"weight": 0},
                }
            }
            features.append(feature)
            point_features.append(point_feature)
            folium.PolyLine(locations=lat_long_coords, color="black", weight=1, tooltip=edge_data).add_to(m)
            # if plot_marker:
            #     folium.Marker(location=lstring.coords[midpoint], popup=f"Edge {count}: {name}", icon=icon_number).add_to(m)
            # count += 1
    folium.plugins.TimestampedGeoJson(
        {
            "type": "FeatureCollection",
            "features": features,
        },
        period="PT1M",
        add_last_point=False,
        # duration="PT5M",
    ).add_to(m)
    folium.plugins.TimestampedGeoJson(
        {
            "type": "FeatureCollection",
            "features": point_features,
        },
        period="PT1M",
        add_last_point=False,
        duration="PT45S",
    ).add_to(m)


    return m