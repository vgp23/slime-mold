from agent import *
from functools import partial
import jax
import jax.numpy as jnp
import jax.random as jr
import collections


Scene = collections.namedtuple('Scene', ['agent_grid', 'trail_grid', 'chemo_grid'])


@partial(jax.jit, static_argnames=['height, width, initial_population_density'])
def scene_init(height, width, initial_population_density, key):
    agent_grid = jnp.zeros((height, width))
    trail_grid = jnp.zeros((height, width))
    chemo_grid = jnp.zeros((height, width))

    return Scene(agent_grid, trail_grid, chemo_grid)
import jax