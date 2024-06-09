from math import atan2
from math import pi

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
