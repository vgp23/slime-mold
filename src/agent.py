from functools import partial
import jax
import jax.random as jr
import jax.numpy as jnp
import collections


@jax.jit
def agent_init(key):
    """Generate a random new agent with [dx, dy, counter]."""

    # use a table because a velocity of 0,0 is not valid, easier to avoid
    # this way
    velocities = jnp.array(
        [(0,1),(1,1),(1,0),(1,-1),(0,-1),(-1,-1),(-1,0),(-1,1)])
    velocity = jr.choice(key, velocities)

    return jnp.array([*velocity, 0])


@partial(jax.jit, static_argnames=['sensor_length'])
def agent_sensor_positions(agent, sensor_length):
    """Compute the relative positions of the sensors using a sensor map."""

    # velocities with 0,0 are not valid; -5 values make this obvious for
    # debbugging (this index should never be reached)
    sensor_map = jnp.array([
        [[(-1,  0), ( 0, -1)], [(-1, -1), ( 1, -1)], [( 0, -1), ( 1,  0)]],
        [[(-1,  1), (-1, -1)], [( -5,-5), ( -5,-5)], [( 1, -1), ( 1,  1)]],
        [[( 0,  1), (-1,  0)], [( 1,  1), (-1,  1)], [( 1,  0), ( 0,  1)]],
    ])
    y = agent[1] + 1
    x = agent[0] + 1
    return sensor_length * jnp.array(sensor_map[y][x])


@jax.jit
def agent_sensor_directions(agent):
    """Compute the directions of the sensors using a sensor map."""

    # velocities with 0,0 are not valid; -5 values make this obvious for
    # debbugging (this index should never be reached)
    sensor_map = jnp.array([
        [[(-1,  0), ( 0, -1)], [(-1, -1), ( 1, -1)], [( 0, -1), ( 1,  0)]],
        [[(-1,  1), (-1, -1)], [( -5,-5), ( -5,-5)], [( 1, -1), ( 1,  1)]],
        [[( 0,  1), (-1,  0)], [( 1,  1), (-1,  1)], [( 1,  0), ( 0,  1)]],
    ])
    y = agent[1] + 1
    x = agent[0] + 1
    return jnp.array(sensor_map[y][x])