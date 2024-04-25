from functools import partial
import jax
import jax.random as jr
import jax.numpy as jnp
import collections


@jax.jit
def agent_init(key):
    """Generate a random new agent with [dx, dy, counter]."""

    def generate_velocity(carry_in):
        _, key = carry_in
        key, subkey = jr.split(key, 2)
        dx, dy = jr.randint(subkey, shape=(2,), minval=-1, maxval=2)
        return ((dx,dy),key)

    def is_not_valid(carry_in):
        ''' Ensures no agents with velocity 0,0 get generated'''
        dx, dy = carry_in[0]
        return  (dx + dy) == 0

    init_input = (0,key)
    carry_out = generate_velocity(init_input)

    velocity, _ = \
        jax.lax.while_loop(is_not_valid, generate_velocity, carry_out)

    return jnp.array([*velocity, 0])


@partial(jax.jit, static_argnames=['sensor_length'])
def agent_sensor_positions(agent, sensor_length):
    """Compute the relative positions of the sensors using a sensor map."""

    # velocities with 0,0 are not valid; -5 values make this obvious for 
    # debbugging (this index should never be reached)
    sensor_map = jnp.array([
        [[(-1,  0), ( 0, -1)], [(-1, -1), ( 1, -1)], [( 0, -1), ( 1,  0)]],
        [[(-1,  1), (-1, -1)], [( -5,  -5), ( -5,  -5)], [( 1, -1), ( 1,  1)]],
        [[( 0,  1), (-1,  0)], [( 1,  1), (-1,  1)], [( 1,  0), ( 0,  1)]],
    ])
    y = agent[1] + 1
    x = agent[0] + 1
    return sensor_length * jnp.array(sensor_map[y][x])
