from functools import partial
import numpy as np
import collections


def agent_init():
    """Generate a random new agent with [dx, dy, counter]."""
    # use a table because a velocity of 0,0 is not valid, easier to avoid this way
    velocities = np.array(
        [(0,1),(1,1),(1,0),(1,-1),(0,-1),(-1,-1),(-1,0),(-1,1)])
    velocity = np.random.choice(velocities)
    return np.array([*velocity, 0])


def agent_sensor_positions(agent, sensor_length):
    """Compute the relative positions of the sensors using a sensor map."""

    # velocities with 0,0 are not valid; -5 values make this obvious for
    # debbugging (this index should never be reached)
    sensor_map = np.array([
        [[(-1,  0), ( 0, -1)], [(-1, -1), ( 1, -1)], [( 0, -1), ( 1,  0)]],
        [[(-1,  1), (-1, -1)], [( -5,-5), ( -5,-5)], [( 1, -1), ( 1,  1)]],
        [[( 0,  1), (-1,  0)], [( 1,  1), (-1,  1)], [( 1,  0), ( 0,  1)]],
    ])
    y = agent[1] + 1
    x = agent[0] + 1
    return sensor_length * np.array(sensor_map[y][x])


def agent_sensor_directions(agent):
    """Compute the directions of the sensors using a sensor map."""

    # velocities with 0,0 are not valid; -5 values make this obvious for
    # debbugging (this index should never be reached)
    sensor_map = np.array([
        [[(-1,  0), ( 0, -1)], [(-1, -1), ( 1, -1)], [( 0, -1), ( 1,  0)]],
        [[(-1,  1), (-1, -1)], [( -5,-5), ( -5,-5)], [( 1, -1), ( 1,  1)]],
        [[( 0,  1), (-1,  0)], [( 1,  1), (-1,  1)], [( 1,  0), ( 0,  1)]],
    ])
    y = agent[1] + 1
    x = agent[0] + 1
    return np.array(sensor_map[y][x])