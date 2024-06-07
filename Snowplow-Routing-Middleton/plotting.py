import itertools as it
import matplotlib.pyplot as plt
import networkx as nx


def draw_labeled_multigraph(G, pos, attr_name=None, ax=None, color='blue', title="Graph Representation of Town", plotDepot=False):
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
    
    # Works with arc3 and angle3 connectionstyles
    connectionstyle = [f"arc3,rad={r}" for r in it.accumulate([0.15] * 4)]
    # connectionstyle = [f"angle3,angleA={r}" for r in it.accumulate([30] * 4)]

    node_lables = dict([(node, "") if node != DEPOT else (node, "depot") for node in G.nodes()]) # label the depot

    plt.figure(figsize=(75, 50))
    plt.title(title, size=50)

    if plotDepot:
        plt.plot(coords['x'], coords['y'], 'ro', label='Depot', markersize=50)

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
        for step in route:
            if G_graph[step.node1][step.node2][step.id].get('order') is None:
                G_graph[step.node1][step.node2][step.id]['order'] = str(count)
            else:
                G_graph[step.node1][step.node2][step.id]['order'] += ", " + str(count)
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