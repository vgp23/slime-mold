from functools import partial
import jax
import jax.random as jr
import jax.numpy as jnp
import collections


@jax.jit
def agent_init(key):
    """Generate a random new agent with [dx, dy, counter]."""
    dx, dy = jr.randint(key, shape=(2,), minval=-1, maxval=2)
    return jnp.array([dx, dy, 0])


@partial(jax.jit, static_argnames=['sensor_length'])
def agent_sensor_positions(agent, sensor_length):
    """Compute the relative positions of the sensors using a sensor map."""
    sensor_map = jnp.array([
        [[(-1,  0), ( 0, -1)], [(-1, -1), ( 1, -1)], [( 0, -1), ( 1,  0)]],
        [[(-1,  1), (-1, -1)], [( 0,  0), ( 0,  0)], [( 1, -1), ( 1,  1)]],
        [[( 0,  1), (-1,  0)], [( 1,  1), (-1,  1)], [( 1,  0), ( 0,  1)]],
    ])
    y = agent[1] + 1
    x = agent[0] + 1
    return sensor_length * jnp.array(sensor_map[y][x])