import jax.random as jr
import jax.numpy as jnp
import collections


Agent = collections.namedtuple('Agent', ['dir', 'count'])
Vec2  = collections.namedtuple('Vec2',  ['x', 'y'])


def agent_init(key):
    """Generate a random new agent."""
    x, y = jr.randint(key, shape=(2,), minval=-1, maxval=2)
    return Agent(Vec2(x, y), 0)


def agent_sensor_positions(agent, sensor_length):
    """Compute the positions of the sensors using a sensor map."""
    sensor_map = [
        [[(-1,  0), ( 0, -1)], [(-1, -1), ( 1, -1)], [( 0, -1), ( 1,  0)]],
        [[(-1,  1), (-1, -1)], [( 0,  0), ( 0,  0)], [( 1, -1), ( 1,  1)]],
        [[( 0,  1), (-1,  0)], [( 1,  1), (-1,  1)], [( 1,  0), ( 0,  1)]],
    ]
    return sensor_map[agent.dir.y + 1][agent.dir.x + 1]
