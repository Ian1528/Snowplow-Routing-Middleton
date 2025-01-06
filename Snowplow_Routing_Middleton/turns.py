from math import atan2
from math import pi
import networkx as nx

def angle_between_points(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]) -> float:
    '''
    Calculates the signed angle in degrees between vectors ab and bc. Negative if bc is clockwise of ab.
    
    Parameters:
        a (tuple): The coordinates of point a.
        b (tuple): The coordinates of point b.
        c (tuple): The coordinates of point c.
    
    Returns:
        float: The signed angle in degrees between vectors ab and bc.
    '''
    v = (b[0]-a[0], b[1]-a[1])
    w = (c[0]-b[0], c[1]-b[1])

    dot = v[0]*w[0] + v[1]*w[1]     # Dot product between [x1, y1] and [x2, y2]
    det = v[0]*w[1] - v[1]*w[0]      # Determinant
    angle = atan2(det, dot)  # atan2(y, x) or atan2(sin, cos)

    return angle*180/pi

def angle_between_vectors(v: tuple, w: tuple) -> float:
    """
    Returns the signed angle (in degrees) between two vectors v and w

    Args:
        v (tuple): A tuple that represetns a vector of the form (x,y)
        w (tuple): The second vector

    Returns:
        float: The signed angle (in degrees) between the two vectors
    """

    dot = v[0]*w[0] + v[1]*w[1]     # Dot product between [x1, y1] and [x2, y2]
    det = v[0]*w[1] - v[1]*w[0]      # Determinant
    angle = atan2(det, dot)  # atan2(y, x) or atan2(sin, cos)

    return angle*180/pi

def turn_direction(angle: float) -> str:
    """
    Returns the direction of a turn given an angle

    Args:
        angle (float): The angle of the turn

    Returns:
        str: A word describing the turn direction (i.e., left, right, straight)
    """

    if angle < 15 and angle > -15:
        return "straight"
    elif angle >= -90 and angle <= -15:
        return "right"
    elif angle <= 90 and angle >= 15:
        return "left"
    elif angle <= 135 and angle > 90:
        return "sharp left"
    elif angle >= -135 and angle < -90:
        return "sharp right"
    else:
        return "u-turn"

def turn_direction_count(G_DUAL: nx.MultiDiGraph, full_route: list[tuple[int, int, int]]) -> tuple[dict[str:int], list[str], list[int]]:
    """
    Calculate the count of different turn directions in a given route.
    Parameters:
    G_DUAL (nx.MultiDiGraph): A directed graph representing the road network.
    full_route (list[tuple[int, int, int]]): A list of tuples representing the route, where each tuple represents an edge.
    Returns:
    tuple: A tuple containing:
        - turns_hist (dict[str:int]): A dictionary with turn directions as keys and their counts as values.
        - x_axis_bins (list[str]): A list of turn direction categories.
        - y_axis (list[int]): A list of counts corresponding to each turn direction category.
    """

    turns_hist = dict()
    for i in range(len(full_route)-1):
        attrb = G_DUAL.get_edge_data(full_route[i], full_route[i+1])
        if attrb is None:
            continue
        if "angle" in attrb[0].keys():
            angle = attrb[0]["angle"]
            turns_hist[turn_direction(angle)] = turns_hist.get(turn_direction(angle), 0) + 1
    x_axis_bins = ["straight", "right", "sharp right", "left", "sharp left", "u-turn"]
    y_axis = [turns_hist.get(turn, 0) for turn in x_axis_bins]
    return turns_hist, x_axis_bins, y_axis