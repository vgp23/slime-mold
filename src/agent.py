from functools import partial
import numpy as np
import collections


dirs = np.array([(-1,-1),(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1)])

def agent_init():
    """Generate a random new agent with [dx, dy, counter]."""
    index = np.random.randint(len(dirs))
    return np.array([*dirs[index], 0])


def agent_sensor_directions(agent):
    """Compute the directions of the sensors using a sensor map."""
    sensor_map = np.array([
        [[(-1,  0), ( 0, -1)], [(-1, -1), ( 1, -1)], [( 0, -1), ( 1,  0)]],
        [[(-1,  1), (-1, -1)], [( -5,-5), ( -5,-5)], [( 1, -1), ( 1,  1)]],
        [[( 0,  1), (-1,  0)], [( 1,  1), (-1,  1)], [( 1,  0), ( 0,  1)]],
        # [[( 0, -1), (-1,  0)], [(-1, -1), (-1,  1)], [(-1,  0), ( 0,  1)]],
        # [[( 1, -1), (-1, -1)], [( -5,-5), ( -5,-5)], [(-1,  1), ( 1,  1)]],
        # [[( 1,  0), ( 0, -1)], [( 1,  1), ( 1, -1)], [( 0,  1), ( 1,  0)]],
    ])
    return sensor_map[*(agent[:2] + 1)]


def agent_sensor_positions(agent, c):
    """Compute the relative positions of the sensors using a sensor map."""
    return c.sensor_length * agent_sensor_directions(agent)