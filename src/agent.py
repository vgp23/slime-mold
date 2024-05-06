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
    # TODO optimise this search away by only saving the index
    # index = next((i for i, t in enumerate(dirs) if np.array_equal(t, agent[:2])), -1)

    # l_sensor = np.array(dirs[(index - 1) % len(dirs)])
    # r_sensor = np.array(dirs[(index + 1) % len(dirs)])

    # return np.array([l_sensor, r_sensor])

    sensor_map = np.array([
        [[( 0, -1), (-1,  0)], [(-1, -1), (-1,  1)], [(-1,  0), ( 0,  1)]],
        [[( 1, -1), (-1, -1)], [( -5,-5), ( -5,-5)], [(-1,  1), ( 1,  1)]],
        [[( 1,  0), ( 0, -1)], [( 1,  1), ( 1, -1)], [( 0,  1), ( 1,  0)]],
    ])
    return sensor_map[*(agent[:2] + 1)]


def agent_sensor_positions(agent, sensor_length):
    """Compute the relative positions of the sensors using a sensor map."""
    return sensor_length * agent_sensor_directions(agent)