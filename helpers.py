import numpy as np
from scipy.stats import norm

# https://stackoverflow.com/questions/2827393/angles-between-two-n-dimensional-vectors-in-python
def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def euclid_dis(from_, to_):
    return np.sqrt(np.square(to_.astype(np.float32) - from_.astype(np.float32)).sum())

def normalized_polygon(vertices, tensor, sd=1):
    def draw_between(from_, to_, tensor):
        # print(f"from {from_} to {to_}")
        distance = euclid_dis(from_, to_)

        vec = to_ - from_
        for i in range(int(distance + 1)):
            current_pos = ((i / distance) * vec + from_).astype(int)
            closest_vert_dist = min(i, distance - i)

            l = norm.pdf(closest_vert_dist, scale=sd)

            tensor[0, np.clip(current_pos[0], 0, 223), np.clip(current_pos[1], 0, 223)] = l

    last_vertex = None
    vertices = vertices.reshape(-1, 2)
    first_vert = np.array([vertices[0][1], vertices[0][0]])
    for vertex in vertices:
        x, y = vertex[1], vertex[0]

        if last_vertex is None:
            last_vertex = np.array([x, y])
            continue 
        
        to_ = np.array([x, y])
        from_ = last_vertex

        draw_between(from_, to_, tensor)

        last_vertex = to_

    draw_between(last_vertex, first_vert, tensor)
    
    return tensor