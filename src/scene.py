from agent import *
from functools import partial
import jax
import jax.numpy as jnp
import jax.random as jr
import collections


@partial(jax.jit, static_argnames=['height', 'width', 'initial_population_density'])
def scene_init(height, width, initial_population_density, key):
    # generate the positions at which we got agents using the specified density
    key, subkey = jr.split(key, 2)
    mask_grid = jr.uniform(subkey, (height, width)) < initial_population_density

    # generate agent data at the positions in mask_grid
    key, *subkeys = jr.split(key, height * width + 1)
    subkeys = jnp.array(subkeys).reshape((height, width, len(key)))
    agent_grid = jax.vmap(jax.vmap(
        lambda mask, key: jax.lax.cond(
            mask, lambda k: agent_init(k), lambda _k: jnp.zeros((3,)).astype(int), key
        )
    ))(mask_grid, subkeys)

    trail_grid = jnp.zeros((height, width))  # contains trail data
    chemo_grid = jnp.zeros((height, width))  # contains chemo-nutrient/food data

    return agent_grid, mask_grid, trail_grid, chemo_grid